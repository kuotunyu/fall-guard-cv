"""fsm.py 單元測試:合成特徵序列驗證狀態機行為(docs/PLAN.md Phase 2 DoD)。"""

from __future__ import annotations

from fallguard.fsm import FallStateMachine, FSMConfig, State


def _frame(t, theta=0.0, omega=0.0, v_y=0.0, rho=0.4, hip_height=0.8, torso_missing=False):
    return {"t": t, "theta": theta, "omega": omega, "v_y": v_y, "rho": rho, "hip_height": hip_height, "torso_missing": torso_missing}


def _run(fsm: FallStateMachine, frames: list[dict]) -> State:
    state = fsm.state
    for f in frames:
        state = fsm.step(f)
    return state


def test_fast_fall_reaches_confirmed_and_alerts():
    """快速下墜(v_y 超閾值連續 2 幀)→ 1 秒內達躺姿 → 躺姿持續 N 秒 → CONFIRMED → ALERTED。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=2.0))
    dt = 1.0 / 25
    frames = []
    t = 0.0
    # 站立一段時間
    for _ in range(25):
        frames.append(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    # 快速下墜:v_y 超過 2.0 torso/s,連續多幀
    for _ in range(5):
        frames.append(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    # 落地躺平,維持超過 confirm_seconds
    for _ in range(75):  # 3 秒份,足夠超過 2 秒確認門檻
        frames.append(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt

    final_state = _run(fsm, frames)
    assert fsm.confirmed_at is not None, "應該要到達 CONFIRMED"
    assert len(fsm.alerts) == 1
    assert not fsm.alerts[0].escalation
    assert final_state == State.ALERTED


def test_slow_lying_down_never_triggers_falling():
    """緩慢躺下(v_y 與 omega 皆低於閾值)→ 不應離開 NORMAL,不告警。"""
    fsm = FallStateMachine()
    dt = 1.0 / 25
    frames = []
    t = 0.0
    # 緩慢從站姿過渡到躺姿,theta 慢慢變大,但 v_y 全程很小(< 2.0 torso/s 閾值)
    for i in range(150):  # 6 秒緩慢躺下
        theta = 5.0 + (80.0 - 5.0) * (i / 150)
        frames.append(_frame(t, theta=theta, v_y=0.3, omega=10.0, rho=0.4 + 1.0 * (i / 150), hip_height=0.9 - 0.7 * (i / 150)))
        t += dt
    # 躺平後維持一段時間
    for _ in range(75):
        frames.append(_frame(t, theta=80.0, v_y=0.0, rho=1.4, hip_height=0.2))
        t += dt

    final_state = _run(fsm, frames)
    assert final_state == State.NORMAL
    assert fsm.confirmed_at is None
    assert len(fsm.alerts) == 0


def test_crouching_does_not_confirm():
    """蹲下:可能短暫觸發 FALLING(下墜速度稍快),但髖高不夠低(未達躺姿),1 秒逾時退回 NORMAL。"""
    fsm = FallStateMachine()
    dt = 1.0 / 25
    frames = []
    t = 0.0
    for _ in range(25):
        frames.append(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    # 蹲下動作:v_y 短暫超標觸發 FALLING,但 hip_height 只降到 0.6(仍 > 0.5 門檻,不算躺姿)
    for _ in range(5):
        frames.append(_frame(t, theta=20.0, v_y=2.5, rho=0.5, hip_height=0.6))
        t += dt
    # 蹲著不動,髖高持續 > 0.5,逾時應退回 NORMAL(1 秒內都不滿足躺姿的三條件)
    for _ in range(50):  # 2 秒份,足夠超過 1 秒逾時窗
        frames.append(_frame(t, theta=25.0, v_y=0.0, rho=0.5, hip_height=0.6))
        t += dt

    final_state = _run(fsm, frames)
    assert final_state == State.NORMAL
    assert fsm.confirmed_at is None
    assert len(fsm.alerts) == 0


def test_brief_posture_jitter_does_not_reset_confirm_timer():
    """ON_GROUND 期間 <=0.5s 的姿勢抖動不應重置躺姿累積計時(仍應到達 CONFIRMED)。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=2.0, lying_jitter_tolerance_s=0.5))
    dt = 1.0 / 25
    t = 0.0
    frames = []
    for _ in range(25):
        frames.append(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    for _ in range(5):
        frames.append(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    # 躺平 0.8 秒
    for _ in range(20):
        frames.append(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    # 短暫抖動 0.2 秒(< 0.5s 容忍上限):姿勢暫時不符合躺姿條件
    for _ in range(5):
        frames.append(_frame(t, theta=50.0, v_y=0.5, rho=0.9, hip_height=0.55))
        t += dt
    # 繼續躺平,累計躺著時間應該持續累加(不重置),很快到達 2 秒總門檻
    for _ in range(40):
        frames.append(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt

    final_state = _run(fsm, frames)
    assert fsm.confirmed_at is not None, "短暫抖動不應阻止 CONFIRMED"


def test_recovery_to_normal_after_standing_up():
    """CONFIRMED 後如果重新站直並維持 2 秒,應恢復 NORMAL。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=1.0))
    dt = 1.0 / 25
    t = 0.0
    frames = []
    for _ in range(25):
        frames.append(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    for _ in range(5):
        frames.append(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    for _ in range(30):  # 足夠達到 confirm_seconds=1.0 並觸發 ALERTED
        frames.append(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    # 站起來,維持 2 秒以上
    for _ in range(60):
        frames.append(_frame(t, theta=5.0, v_y=0.0, rho=0.35, hip_height=0.9))
        t += dt

    final_state = _run(fsm, frames)
    assert fsm.confirmed_at is not None
    assert final_state == State.NORMAL, "重新站直 2 秒後應恢復 NORMAL"


def test_high_missing_rate_freezes_state_machine():
    """軀幹缺失率 > 50% 時應凍結,不因缺失資料而錯誤轉移。"""
    fsm = FallStateMachine(FSMConfig(torso_missing_freeze_window_s=1.0, torso_missing_freeze_ratio=0.5))
    dt = 1.0 / 25
    t = 0.0
    setup_frames = []
    for _ in range(25):
        setup_frames.append(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    # 觸發到 ON_GROUND
    for _ in range(5):
        setup_frames.append(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    for _ in range(20):
        setup_frames.append(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt

    state_before_occlusion = _run(fsm, setup_frames)
    assert state_before_occlusion == State.ON_GROUND

    # 家具遮擋:軀幹大量缺失,持續超過凍結視窗
    occlusion_frames = []
    for _ in range(30):
        occlusion_frames.append(_frame(t, torso_missing=True))
        t += dt

    final_state = _run(fsm, occlusion_frames)
    assert final_state == State.ON_GROUND, "缺失率過高時應凍結在原狀態,不應被重置回 NORMAL"


def test_lying_elapsed_s_tracks_on_ground_countdown():
    """`lying_elapsed_s` 供 detect.py 疊加 ON_GROUND 倒數用:NORMAL 時為 None,進 ON_GROUND 後隨時間累加。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=2.0))
    dt = 1.0 / 25
    t = 0.0
    assert fsm.lying_elapsed_s is None

    for _ in range(25):
        fsm.step(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    assert fsm.lying_elapsed_s is None, "NORMAL 狀態不應有躺姿累積計時"

    for _ in range(5):
        fsm.step(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    assert fsm.state == State.FALLING
    assert fsm.lying_elapsed_s is None, "FALLING 尚未進 ON_GROUND,不應有躺姿累積計時"

    fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
    t += dt
    assert fsm.state == State.ON_GROUND
    assert fsm.lying_elapsed_s == 0.0, "剛進 ON_GROUND 的第一幀,累積時間應為 0"

    for _ in range(12):  # 再躺 0.48s,尚未達 confirm_seconds=2.0
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    assert fsm.state == State.ON_GROUND
    assert fsm.lying_elapsed_s is not None and 0.4 < fsm.lying_elapsed_s < 0.6


def test_on_ground_lying_timer_continues_through_partial_feature_occlusion():
    """D16:軀幹可見(不觸發 `_is_frozen` 凍結)但其他特徵局部缺失(NaN)時,純時間的躺姿
    累積計時仍要照常檢查——先前只有 test_high_missing_rate_freezes_state_machine 測到
    「凍結」分支(torso_missing=True 直接 return,根本不會走到這段邏輯),這條路徑先前
    完全沒有測試覆蓋過(收尾複查發現)。缺失時間 <= jitter 容忍上限:計時不中斷。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=2.0, lying_jitter_tolerance_s=0.5))
    dt = 1.0 / 25
    t = 0.0
    for _ in range(25):
        fsm.step(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    for _ in range(5):
        fsm.step(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    for _ in range(20):  # 躺平 0.8 秒,累積躺姿計時開始
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    assert fsm.state == State.ON_GROUND

    # 局部遮擋 0.2 秒(< 0.5s 容忍上限):torso_missing=False(不凍結),但其他特徵 NaN
    for _ in range(5):
        t += dt
        fsm.step(_frame(t, theta=float("nan"), v_y=float("nan"), rho=float("nan"), hip_height=float("nan")))

    # 繼續躺平,累計時間應該持續往前走(未被 NaN 中斷),很快到達 2 秒總門檻
    for _ in range(40):
        t += dt
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))

    assert fsm.confirmed_at is not None, "局部遮擋(<=容忍上限)不應阻止 CONFIRMED"


def test_on_ground_lying_timer_resets_after_prolonged_occlusion():
    """D16 對照組:局部遮擋超過 jitter 容忍上限時,躺姿累積計時應重新起算(不是無限容忍)。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=2.0, lying_jitter_tolerance_s=0.5))
    dt = 1.0 / 25
    t = 0.0
    for _ in range(25):
        fsm.step(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    for _ in range(5):
        fsm.step(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    for _ in range(20):  # 躺平 0.8 秒
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    assert fsm.state == State.ON_GROUND
    elapsed_before = fsm.lying_elapsed_s

    # 局部遮擋 0.8 秒(> 0.5s 容忍上限)
    for _ in range(20):
        t += dt
        fsm.step(_frame(t, theta=float("nan"), v_y=float("nan"), rho=float("nan"), hip_height=float("nan")))

    # 遮擋結束,恢復躺姿讀數:累積計時應已重新起算,不是延續遮擋前的累積時間
    t += dt
    fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
    assert fsm.state == State.ON_GROUND
    assert fsm.lying_elapsed_s is not None and fsm.lying_elapsed_s < elapsed_before


def test_escalation_alert_fires_after_cooldown_while_still_on_ground():
    """冷卻時間(cooldown_s)過後仍倒地,應觸發第二個 escalation=True 的升級再告警;
    冷卻時間內不應重複告警。這條分支(fsm.py `elif self.state == State.ALERTED`)先前
    完全沒有測試觸發過或斷言過(收尾複查發現)。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=0.5, cooldown_s=1.0))
    dt = 1.0 / 25
    t = 0.0
    for _ in range(25):
        fsm.step(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    for _ in range(5):
        fsm.step(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    for _ in range(20):  # 躺平 0.8 秒,足夠達到 confirm_seconds=0.5 並觸發 ALERTED
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    assert fsm.state == State.ALERTED
    assert len(fsm.alerts) == 1
    assert fsm.alerts[0].escalation is False

    # 冷卻時間內(累計未超過 1.0s)持續躺著,不應重複告警
    for _ in range(15):  # 0.6s
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    assert len(fsm.alerts) == 1, "冷卻時間內不應重複告警"

    # 繼續躺著直到累計超過 cooldown_s,應觸發升級再告警
    for _ in range(15):  # 再 0.6s
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    assert len(fsm.alerts) == 2
    assert fsm.alerts[1].escalation is True


def test_occluded_recovery_does_not_silently_revert_alerted_to_normal():
    """D49:恢復計時在特徵局部缺失(遮擋)期間不應讓空白時間也算數——否則等於在沒有持續
    視覺證據下靜默撤銷已確認的告警。修正前:1 幀疑似站起的證據 + 之後任意長度的遮擋,
    只要牆鐘時間累積滿 recovery_hold_s 就會誤判為已恢復;修正後遮擋期間應重新起算。"""
    fsm = FallStateMachine(FSMConfig(confirm_seconds=0.5, cooldown_s=1000.0, recovery_jitter_tolerance_s=0.5))
    dt = 1.0 / 25
    t = 0.0
    for _ in range(25):
        fsm.step(_frame(t, theta=5.0, v_y=0.1, rho=0.35, hip_height=0.9))
        t += dt
    for _ in range(5):
        fsm.step(_frame(t, theta=30.0, v_y=3.0, rho=0.6, hip_height=0.6))
        t += dt
    for _ in range(20):
        fsm.step(_frame(t, theta=80.0, v_y=0.0, rho=1.5, hip_height=0.2))
        t += dt
    assert fsm.state == State.ALERTED

    # 1 幀符合恢復條件(疑似站起),設下 _recovery_start_t
    t += dt
    fsm.step(_frame(t, theta=10.0, v_y=0.0, rho=0.4, hip_height=0.8))
    assert fsm._recovery_start_t is not None

    # 之後連續 3 秒遮擋:軀幹可見(torso_missing=False,不觸發凍結)但其他特徵缺失
    for _ in range(75):
        t += dt
        fsm.step(_frame(t, theta=float("nan"), v_y=0.0, rho=float("nan"), hip_height=float("nan")))

    assert fsm.state == State.ALERTED, "遮擋期間不應被靜默判定為已恢復直立而撤銷告警"
