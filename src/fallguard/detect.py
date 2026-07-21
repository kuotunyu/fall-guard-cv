"""跌倒偵測即時推論 CLI(docs/PLAN.md §8.4)。

三執行緒:
- capture thread:`cv2.VideoCapture` 連續讀取,1-slot 佇列只留最新幀(webcam 驅動有
  內部緩衝,直接 read 會累積延遲,官方常見作法是丟棄舊幀只處理最新的)
- main thread:`model.track()` 推論 → 特徵計算 → 狀態機 → overlay → imshow。特徵計算
  重用 features.py 對一段滑動緩衝區重跑既有的批次邏輯,不另外維護一份增量版特徵計算
  (避免評估與部署兩套特徵邏輯飄移,見 D18 的教訓)
- alert worker(`ThreadPoolExecutor(max_workers=1)`):CONFIRMED/冷卻後升級告警時,
  非同步呼叫 VLM→Discord,不阻塞主偵測迴圈

用法:
    uv run python -m fallguard.detect --source data/raw/urfd/fall-01-cam0.mp4
    uv run python -m fallguard.detect --source 0
    uv run python -m fallguard.detect --source <mp4> --no-display --dump-features out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import REPO_ROOT, settings
from .features import compute_features
from .fsm import FallStateMachine, FSMConfig, State
from .pose import PoseEstimator

EVENTS_DIR = REPO_ROOT / "events"

# BGR(cv2 色彩順序),對應 docs/PLAN.md §3 狀態顏色表(綠/黃/橙/紅/紅)
STATE_COLORS = {
    State.NORMAL: (0, 255, 0),
    State.FALLING: (0, 255, 255),
    State.ON_GROUND: (0, 140, 255),
    State.CONFIRMED: (0, 0, 255),
    State.ALERTED: (0, 0, 255),
}

FEATURE_BUFFER_S = 6.0  # compute_features 重採樣/滑動中位数用的回顧視窗(§7.3 最長用到 3s,留一倍餘裕)


class RollingFps:
    """畫面上顯示的 FPS 用「近期」滑動視窗算,不是「累積自程式啟動」的平均——後者會被
    GPU 第一次推論的暖機成本(CUDA kernel 編譯/cudnn 演算法搜尋,一次性、跟穩態吞吐量
    無關)拖著,要等很久才爬升到真實穩態值,使用者一開始看到的數字沒有代表性。近期滑動
    視窗只反映「現在」跑多快,啟動後幾幀內就能穩定顯示正確數值。"""

    def __init__(self, window_s: float = 1.5):
        self.window_s = window_s
        self._timestamps: deque[float] = deque()

    def tick(self, now: float) -> float:
        self._timestamps.append(now)
        while len(self._timestamps) > 1 and now - self._timestamps[0] > self.window_s:
            self._timestamps.popleft()
        if len(self._timestamps) < 2:
            return 0.0
        return (len(self._timestamps) - 1) / (self._timestamps[-1] - self._timestamps[0])


class LatestFrameQueue:
    """1-slot 佇列:capture thread 一直塞最新幀,main thread 只拿最新的,舊幀直接丟棄不排隊。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._closed = False

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get_latest(self, last_seen_id: int) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._frame is None or self._frame_id == last_seen_id:
                return None, last_seen_id
            return self._frame, self._frame_id

    def close(self) -> None:
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed


def capture_loop(
    cap: cv2.VideoCapture,
    queue: LatestFrameQueue,
    stop_event: threading.Event,
    target_dt: float | None,
) -> None:
    """`target_dt` 非 None 時,依來源原生 fps 間隔配速讀取(模擬即時攝影機)——否則影片檔
    會被本執行緒瞬間讀完,1-slot 佇列只留最後一幀,main thread 完全來不及看到中間畫面
    (webcam 天生由硬體配速,不需要這個機制)。吞吐量測試改用完全不同的 `run_benchmark()`
    單執行緒迴圈,不會呼叫到這個函式。"""
    next_t = time.monotonic()
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            queue.close()
            return
        queue.put(frame)
        if target_dt:
            next_t += target_dt
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()  # 已落後就重新起算基準,避免無止盡追趕延遲


def save_snapshot(frame: np.ndarray, tag: str) -> Path:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = EVENTS_DIR / f"{ts}_{tag}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def run_alert(confirm_path: Path, escalation: bool) -> None:
    """alert worker 執行緒的工作內容:VLM 描述(LOCAL_ONLY 時跳過)→ Discord 送出(D8/§8.2/§8.3)。"""
    from . import notify, vlm

    if settings.local_only:
        description = "(LOCAL_ONLY 模式,已跳過雲端 VLM 描述,僅本機留存截圖)"
    else:
        print("[alert] 呼叫 VLM 描述現場...")
        description = vlm.describe_scene(confirm_path)
        print(f"[alert] VLM 描述:{description}")

    print("[alert] 送出 Discord 通報...")
    ok = notify.send_fall_alert(description, image_path=confirm_path, escalation=escalation)
    print(f"[alert] {'已送達' if ok else '送出失敗,詳見上方訊息'}")


def print_cost_estimate() -> None:
    print("=== VLM 呼叫成本估算(docs/PLAN.md 第 11 章) ===")
    print(f"模型:{settings.gemini_model}")
    print("每次通報 ≈ 1 張 720p JPEG + 短 prompt + ~150 token 輸出,單次成本遠低於 $0.001")
    hourly_cap = 3600 / settings.alert_cooldown_seconds if settings.alert_cooldown_seconds > 0 else float("inf")
    print(f"冷卻 {settings.alert_cooldown_seconds:.0f}s ⇒ 每小時最多 {hourly_cap:.0f} 次告警(天花板情境,非預期實際頻率)")
    print("實際單價請以官方定價頁為準;LOCAL_ONLY=true 可完全跳過此項花費")
    print()


def _translucent_rect(frame: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], alpha: float = 0.65) -> None:
    """半透明黑底(比純黑實心矩形不那麼搶畫面),原地疊在 frame 上。"""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def _put_text_outlined(frame: np.ndarray, text: str, org: tuple[int, int], font_scale: float, color: tuple[int, int, int], thickness: int) -> None:
    """疊字文字——每個元素底下都已經有底色矩形襯著,不需要再幫文字本身加黑色描邊
    (描邊會讓字看起來像帶陰影,觀感不舒服,使用者回饋後拿掉);LINE_AA 只是讓邊緣平滑。"""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def overlay_frame(
    frame: np.ndarray,
    state: State,
    feats: dict | None,
    fps: float,
    lying_elapsed: float | None,
    confirm_seconds: float,
) -> np.ndarray:
    color = STATE_COLORS[state]
    label = state.value
    if state == State.ON_GROUND and lying_elapsed is not None:
        label = f"{state.value} {lying_elapsed:.1f}/{confirm_seconds:.1f}s"

    # 底色矩形尺寸只依「畫面高度」縮放(不依當下文字內容),避免兩個問題:
    # (1) 固定絕對像素在低解析度來源(例如 demo 用的 URFD 240p 半格畫面)上佔比過大,
    #     喧賓奪主;(2) 同一畫面緩衝區連續疊字時,矩形若依文字動態縮小會蓋不到
    #     前一幀較寬的殘留文字(見 D23 demo 錄製時的教訓)。特徵讀數併成一行,
    #     大幅縮短原本 3 行文字佔用的高度。
    # 字級各自獨立設下限(不是共用同一個 scale 再往下乘)——例如 0.42*scale 這種
    # 二次縮放,在 scale 已經是下限的窄畫面上會把字體壓到肉眼幾乎讀不出來。
    scale = max(0.5, min(1.4, frame.shape[0] / 480))
    label_font = max(0.55, min(1.15, 0.85 * scale))  # 左上狀態標籤是最關鍵資訊,字級加大
    feat_font = max(0.4, min(0.7, 0.5 * scale))  # 特徵讀數移到右上,字級也順便加大一點
    fps_font = max(0.35, min(0.6, 0.5 * scale))

    # 底色寬度用「最長預期字樣」在該字級下量出來的實際寬度決定(不是量當下這幀的文字),
    # 字級只依畫面尺寸而定、跟目前顯示什麼內容無關,所以每幀量出來的寬度都一樣寬,
    # 不會有 D23 那種「換一幀文字變短、矩形跟著縮小蓋不到殘留字」的問題。
    # 左上(狀態)/右上(特徵+FPS)是兩個獨立的資訊框,寬度各自封頂在畫面寬度的 48%,
    # 確保中間永遠留有間隔——不然文字長一點,兩個底色矩形會頭尾相接,看起來像
    # 融成一整條怪異的通欄黑帶(使用者實測回饋)。
    max_box_w = int(frame.shape[1] * 0.48)

    label_thick = max(1, round(2 * scale))
    label_text_w, label_text_h = cv2.getTextSize("ON_GROUND 10.0/10.0s", cv2.FONT_HERSHEY_SIMPLEX, label_font, label_thick)[0]
    label_h = label_text_h + int(18 * scale)
    label_w = min(max_box_w, label_text_w + int(16 * scale))
    _translucent_rect(frame, (0, 0), (label_w, label_h), alpha=0.65)
    _put_text_outlined(frame, label, (int(8 * scale), int(label_h * 0.72)), label_font, color, label_thick)

    # 右上角合併特徵讀數(第一行)+ FPS(第二行)成一個框,底部因此完全淨空,
    # 也讓「兩個獨立資訊框」的版面更明確,不會有右上/右下兩塊各自為政的感覺。
    if feats is not None:
        feat_line = f"t={feats.get('theta', float('nan')):.1f} v={feats.get('v_y', float('nan')):.2f} h={feats.get('hip_height', float('nan')):.2f}"
    else:
        feat_line = "(no detection yet)"
    fps_line = f"FPS {fps:.1f}"

    feat_text_w, feat_text_h = cv2.getTextSize("t=-180.0 v=-9.99 h=-9.99", cv2.FONT_HERSHEY_SIMPLEX, feat_font, 1)[0]
    fps_text_w, fps_text_h = cv2.getTextSize("FPS 999.9", cv2.FONT_HERSHEY_SIMPLEX, fps_font, 1)[0]
    right_w = min(max_box_w, max(feat_text_w, fps_text_w) + int(14 * scale))
    line_gap = int(6 * scale)
    right_h = feat_text_h + fps_text_h + line_gap + int(20 * scale)

    _translucent_rect(frame, (frame.shape[1] - right_w, 0), (frame.shape[1], right_h), alpha=0.65)
    feat_y = feat_text_h + int(8 * scale)
    _put_text_outlined(frame, feat_line, (frame.shape[1] - right_w + int(7 * scale), feat_y), feat_font, (255, 255, 255), 1)
    fps_y = feat_y + fps_text_h + line_gap
    _put_text_outlined(frame, fps_line, (frame.shape[1] - right_w + int(7 * scale), fps_y), fps_font, (255, 255, 255), 1)
    return frame


def run_benchmark(source: str, pose: PoseEstimator, fsm_config: FSMConfig, min_frames: int = 300) -> None:
    """單執行緒逐幀跑,不透過 1-slot 佇列/capture thread——短片會被那套「只留最新幀」設計瞬間
    讀完丟棄,量不到真實吞吐量(見 D20:webcam 場景丟舊幀是對的,但拿它測固定長度檔案的吞吐量
    不成立)。影片放完就繞回開頭重播,湊到 `min_frames` 才停,避免短片樣本數太少導致量測不穩定。
    量測範圍含 pose 推論+特徵計算+狀態機,不含 imshow(對應 DoD『平均 FPS,目標 ≥30』)。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"無法開啟來源:{source}")
        sys.exit(1)

    fsm = FallStateMachine(fsm_config)
    raw_buf: deque = deque()
    n_frames = 0
    t_start = time.monotonic()

    while n_frames < min_frames:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        t = time.monotonic() - t_start
        pose_frame, _ = pose.infer(frame)
        raw_buf.append((t, pose_frame.xyn, pose_frame.conf, pose_frame.bbox_xywh))
        while raw_buf and t - raw_buf[0][0] > FEATURE_BUFFER_S:
            raw_buf.popleft()

        if len(raw_buf) >= 2:
            ts = np.array([r[0] for r in raw_buf], dtype=np.float64)
            xyn = np.stack([r[1] for r in raw_buf])
            conf = np.stack([r[2] for r in raw_buf])
            bbox = np.stack([r[3] for r in raw_buf])
            feats = compute_features(xyn, conf, bbox, ts)
            if len(feats) > 0:
                fsm.step(feats.frame(-1))

        n_frames += 1

    cap.release()
    elapsed = time.monotonic() - t_start
    avg_fps = n_frames / elapsed if elapsed > 0 else 0.0
    print(f"[benchmark] 共處理 {n_frames} 幀,耗時 {elapsed:.2f}s,平均 FPS={avg_fps:.1f}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="影片路徑,或 webcam 索引(例如 0)")
    parser.add_argument("--confirm-seconds", type=float, default=None, help="覆寫 FALL_CONFIRM_SECONDS(預設讀 .env,部署建議 10)")
    parser.add_argument("--no-display", action="store_true", help="不開 imshow 視窗(headless/量測 FPS 用)")
    parser.add_argument("--dump-features", type=Path, default=None, help="把逐幀特徵+狀態寫成 CSV,供除錯/測試 fixture")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="單執行緒吞吐量測試(不透過即時 1-slot 佇列),量測管線真實 FPS 上限,驗收 DoD ≥30 FPS 用;僅支援影片檔來源",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.lstrip("-").isdigit() else args.source
    is_file_source = isinstance(source, str)

    if args.benchmark:
        if not is_file_source:
            print("--benchmark 僅支援影片檔來源(webcam 沒有『讀完一輪』的概念)")
            sys.exit(1)
        print("載入 pose 模型...")
        pose = PoseEstimator(settings.pose_model)
        fsm_config = FSMConfig(
            confirm_seconds=args.confirm_seconds if args.confirm_seconds is not None else settings.fall_confirm_seconds,
            cooldown_s=settings.alert_cooldown_seconds,
        )
        run_benchmark(source, pose, fsm_config)
        return

    if not settings.local_only:
        print_cost_estimate()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"無法開啟來源:{args.source}")
        sys.exit(1)

    target_dt = None
    if is_file_source:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_dt = 1.0 / native_fps
        print(f"影片來源配速至原生 {native_fps:.1f} FPS(模擬即時攝影機;量測吞吐量請改用 --benchmark)")

    # 模型載入(含 CUDA 暖機)先做完,才啟動 capture thread——否則配速讀取的影片檔
    # 會在暖機期間被悄悄讀完丟棄(1-slot 佇列只留最新幀),白白浪費整段短片。
    print("載入 pose 模型...")
    pose = PoseEstimator(settings.pose_model)
    fsm_config = FSMConfig(
        confirm_seconds=args.confirm_seconds if args.confirm_seconds is not None else settings.fall_confirm_seconds,
        cooldown_s=settings.alert_cooldown_seconds,
    )
    fsm = FallStateMachine(fsm_config)

    queue = LatestFrameQueue()
    stop_event = threading.Event()
    cap_thread = threading.Thread(target=capture_loop, args=(cap, queue, stop_event, target_dt), daemon=True)
    cap_thread.start()

    raw_buf: deque = deque()  # 每格 (t, xyn, conf, bbox_xywh)
    impact_frame: np.ndarray | None = None

    alert_pool = ThreadPoolExecutor(max_workers=1)
    n_alerts_seen = 0

    dump_rows: list[dict] = [] if args.dump_features else None

    t_start = time.monotonic()
    n_frames = 0
    last_seen_id = 0
    fps_tracker = RollingFps()

    print("偵測中...(視窗按 q 結束;headless 模式 Ctrl+C 結束)")
    try:
        while True:
            frame, last_seen_id = queue.get_latest(last_seen_id)
            if frame is None:
                if queue.closed:
                    break
                time.sleep(0.001)
                continue

            t = time.monotonic() - t_start
            pose_frame, annotated = pose.infer(frame)

            raw_buf.append((t, pose_frame.xyn, pose_frame.conf, pose_frame.bbox_xywh))
            while raw_buf and t - raw_buf[0][0] > FEATURE_BUFFER_S:
                raw_buf.popleft()

            feats_dict = None
            if len(raw_buf) >= 2:
                ts = np.array([r[0] for r in raw_buf], dtype=np.float64)
                xyn = np.stack([r[1] for r in raw_buf])
                conf = np.stack([r[2] for r in raw_buf])
                bbox = np.stack([r[3] for r in raw_buf])
                feats = compute_features(xyn, conf, bbox, ts)
                if len(feats) > 0:
                    feats_dict = feats.frame(-1)

            prev_state = fsm.state
            if feats_dict is not None:
                fsm.step(feats_dict)
            new_state = fsm.state

            if prev_state == State.FALLING and new_state == State.ON_GROUND:
                impact_frame = frame.copy()

            if len(fsm.alerts) > n_alerts_seen:
                new_alert = fsm.alerts[-1]
                n_alerts_seen = len(fsm.alerts)
                confirm_path = save_snapshot(frame, "confirm")
                if impact_frame is not None:
                    save_snapshot(impact_frame, "impact")
                print(f"[detect] {'升級再告警' if new_alert.escalation else '確認跌倒'} @ t={t:.1f}s → {confirm_path.name}")
                alert_pool.submit(run_alert, confirm_path, new_alert.escalation)

            n_frames += 1
            fps = fps_tracker.tick(time.monotonic())

            if dump_rows is not None and feats_dict is not None:
                dump_rows.append({"t": t, "state": new_state.value, **{k: v for k, v in feats_dict.items() if k != "t"}})

            if not args.no_display:
                out = overlay_frame(annotated, new_state, feats_dict, fps, fsm.lying_elapsed_s, fsm_config.confirm_seconds)
                cv2.imshow("fall-guard-cv", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        alert_pool.shutdown(wait=True)

    elapsed = time.monotonic() - t_start
    avg_fps = n_frames / elapsed if elapsed > 0 else 0.0
    print(f"結束:共 {n_frames} 幀,耗時 {elapsed:.1f}s,平均 FPS={avg_fps:.1f}")

    if dump_rows is not None:
        with args.dump_features.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(dump_rows[0].keys()) if dump_rows else ["t", "state"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dump_rows)
        print(f"特徵已寫入 {args.dump_features}")


if __name__ == "__main__":
    main()
