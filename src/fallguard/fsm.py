"""跌倒判定狀態機(docs/PLAN.md §8.1)。純函式類,輸入逐幀特徵、輸出狀態+轉移日誌,
不含任何 I/O(截圖/VLM/Discord 由 Phase 4 的 notify.py 呼叫端處理)。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


class State(str, Enum):
    NORMAL = "NORMAL"
    FALLING = "FALLING"
    ON_GROUND = "ON_GROUND"
    CONFIRMED = "CONFIRMED"
    ALERTED = "ALERTED"


@dataclass
class FSMConfig:
    v_y_threshold: float = 2.0  # torso/s(≈1.0 m/s)
    omega_threshold: float = 120.0  # 度/秒
    trigger_min_consecutive_frames: int = 2

    falling_timeout_s: float = 1.0
    on_ground_theta_threshold: float = 60.0
    on_ground_rho_threshold: float = 1.0
    on_ground_hip_height_threshold: float = 0.5

    confirm_seconds: float = 2.0  # N;評估用 2s,部署用 10s(D11,由呼叫端依情境指定)
    lying_jitter_tolerance_s: float = 0.5  # 允許姿勢抖動的最長間斷

    recovery_theta_threshold: float = 40.0  # 進 60°/出 40° 遲滯
    recovery_hip_height_threshold: float = 0.7
    recovery_hold_s: float = 2.0

    cooldown_s: float = 120.0

    torso_missing_freeze_window_s: float = 1.0
    torso_missing_freeze_ratio: float = 0.5


@dataclass
class Transition:
    t: float
    from_state: State
    to_state: State
    reason: str


@dataclass
class AlertEvent:
    t: float
    escalation: bool  # False=首次告警,True=冷卻結束仍倒地的升級再告警


class FallStateMachine:
    """`step(frame)` 逐幀餵入(frame 為 features.FrameFeatures.frame(i) 的字典),
    回傳目前狀態。轉移紀錄存在 `.log`,告警事件存在 `.alerts`。
    """

    def __init__(self, config: FSMConfig | None = None):
        self.config = config or FSMConfig()
        self.state = State.NORMAL
        self.log: list[Transition] = []
        self.alerts: list[AlertEvent] = []

        self._trigger_count = 0
        self._falling_entered_t: float | None = None
        self._lying_accum_start_t: float | None = None
        self._last_lying_true_t: float | None = None
        self._recovery_start_t: float | None = None
        self._last_alert_t: float | None = None
        self._missing_history: deque[tuple[float, bool]] = deque()

    @property
    def confirmed_at(self) -> float | None:
        for tr in self.log:
            if tr.to_state == State.CONFIRMED:
                return tr.t
        return None

    def _is_frozen(self, t: float, torso_missing: bool) -> bool:
        cfg = self.config
        self._missing_history.append((t, torso_missing))
        cutoff = t - cfg.torso_missing_freeze_window_s
        while self._missing_history and self._missing_history[0][0] < cutoff:
            self._missing_history.popleft()
        if not self._missing_history:
            return False
        ratio = sum(1 for _, m in self._missing_history if m) / len(self._missing_history)
        return ratio > cfg.torso_missing_freeze_ratio

    def _transition(self, t: float, to_state: State, reason: str) -> None:
        self.log.append(Transition(t=t, from_state=self.state, to_state=to_state, reason=reason))
        self.state = to_state

    def _lying_criteria(self, theta: float, rho: float, hip_height: float) -> bool:
        cfg = self.config
        return theta > cfg.on_ground_theta_threshold and rho > cfg.on_ground_rho_threshold and hip_height < cfg.on_ground_hip_height_threshold

    def _recovery_criteria(self, theta: float, hip_height: float) -> bool:
        cfg = self.config
        return theta < cfg.recovery_theta_threshold and hip_height > cfg.recovery_hip_height_threshold

    def step(self, frame: dict) -> State:
        """D16 修正：純時間的轉移判斷(FALLING 逾時、ON_GROUND 確認、ALERTED 冷卻/恢復)
        必須在特徵缺失(NaN)時仍照常檢查——否則跌倒瞬間常見的短暫遮擋會讓狀態機永久卡住
        (曾經真實發生：因為早期版本『任一特徵 NaN 就整幀跳過』,連逾時退回都被跳過,
        FALLING 狀態卡死不動,事件級 Sensitivity 因此變成 0)。只有『需要當下特徵值
        才能判斷』的部分(觸發偵測、躺姿判定、恢復判定)才在特徵缺失時跳過。
        """
        cfg = self.config
        t = frame["t"]
        torso_missing = bool(frame.get("torso_missing", False))

        if self._is_frozen(t, torso_missing):
            return self.state  # 缺失率過高:凍結,不轉移不重置(人可能被家具遮擋)

        theta, omega, v_y = frame.get("theta"), frame.get("omega"), frame.get("v_y")
        rho, hip_height = frame.get("rho"), frame.get("hip_height")
        feats_valid = not any(_is_nan(v) for v in (theta, omega, v_y, rho, hip_height))

        if self.state == State.NORMAL:
            if feats_valid:
                triggered = v_y > cfg.v_y_threshold or abs(omega) > cfg.omega_threshold
                self._trigger_count = self._trigger_count + 1 if triggered else 0
                if self._trigger_count >= cfg.trigger_min_consecutive_frames:
                    self._falling_entered_t = t
                    self._trigger_count = 0
                    self._transition(t, State.FALLING, f"v_y={v_y:.2f} omega={omega:.1f} 連續觸發")

        elif self.state == State.FALLING:
            if feats_valid and self._lying_criteria(theta, rho, hip_height):
                self._lying_accum_start_t = t
                self._last_lying_true_t = t
                self._transition(t, State.ON_GROUND, f"theta={theta:.1f} rho={rho:.2f} hip_height={hip_height:.2f}")
            elif t - self._falling_entered_t > cfg.falling_timeout_s:
                self._transition(t, State.NORMAL, "1.0s 內未達躺姿,逾時退回(純時間判斷,不受特徵缺失影響)")

        elif self.state in (State.ON_GROUND, State.CONFIRMED, State.ALERTED):
            self._step_lying_or_recovery(t, theta, rho, hip_height, feats_valid)

        return self.state

    def _step_lying_or_recovery(self, t: float, theta: float, rho: float, hip_height: float, feats_valid: bool) -> None:
        cfg = self.config

        if feats_valid:
            if self._lying_criteria(theta, rho, hip_height):
                if self._lying_accum_start_t is None:
                    self._lying_accum_start_t = t
                self._last_lying_true_t = t
                self._recovery_start_t = None
            else:
                if self._last_lying_true_t is not None and (t - self._last_lying_true_t) > cfg.lying_jitter_tolerance_s:
                    self._lying_accum_start_t = None  # 抖動超過容忍上限,重新起算連續躺著時間

            if self._recovery_criteria(theta, hip_height):
                if self._recovery_start_t is None:
                    self._recovery_start_t = t
            else:
                self._recovery_start_t = None
        else:
            # 特徵缺失:躺著累積計時繼續往前走(不因缺值本身視為抖動中斷),
            # 但缺失時間一旦超過抖動容忍上限,一樣要重新起算(呼應 §7.3 缺失處理原則)。
            if self._last_lying_true_t is not None and (t - self._last_lying_true_t) > cfg.lying_jitter_tolerance_s:
                self._lying_accum_start_t = None

        # 以下皆為純時間判斷,即使本幀特徵缺失(feats_valid=False)也要照常檢查,
        # 否則跌倒後續的短暫遮擋會讓 ON_GROUND→CONFIRMED、ALERTED 冷卻卡死。
        if self._recovery_start_t is not None and t - self._recovery_start_t >= cfg.recovery_hold_s:
            self._transition(t, State.NORMAL, "恢復直立達 2s")
            self._lying_accum_start_t = None
            self._last_lying_true_t = None
            self._recovery_start_t = None
            self._last_alert_t = None
            return

        if self.state == State.ON_GROUND and self._lying_accum_start_t is not None:
            if t - self._lying_accum_start_t >= cfg.confirm_seconds:
                self._transition(t, State.CONFIRMED, f"躺姿持續達 {cfg.confirm_seconds}s")
                self._fire_alert(t, escalation=False)
                self._transition(t, State.ALERTED, "截圖→VLM→Discord")

        elif self.state == State.ALERTED:
            if self._last_alert_t is not None and (t - self._last_alert_t) > cfg.cooldown_s:
                self._fire_alert(t, escalation=True)

    def _fire_alert(self, t: float, escalation: bool) -> None:
        self.alerts.append(AlertEvent(t=t, escalation=escalation))
        self._last_alert_t = t
