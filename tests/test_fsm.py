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
