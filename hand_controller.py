"""
hand_controller.py
Amazing Hand 컨트롤러 (Pollen Robotics 공식 SDK 기반)
=======================================================

공식 SDK
--------
  pip install rustypot
  → Scs0009PyController 로 Feetech SCS 버스 제어
  → Amazing_Hand_Demo.py (공식 예제) 와 동일 패턴

클래스
------
  AmazingHandController
    .connect() / .disconnect()
    .home()      — 전체 손가락 펼침
    .hover()     — 검지만 편 Index-Pointing 자세
    .sniff()     — 가스 센서 흡입 자세 (hover 와 동일)
    .tap()       — phase 전체 동안 반복 tap
    .press()     — 단일 긴 press + hold + release
    .rub()       — 검지 각도 진동 (shear-cue 근사)
    .safe()      — 실험 종료 시 손가락 접기

설계 원칙
---------
  1. 공식 SDK 외부 의존성 없음 (rustypot 만 사용)
  2. 오른손/왼손 좌우 대칭은 Side 파라미터로 자동 처리
  3. 모든 동작 메서드는 동작 완료 후 반환 (blocking)
  4. 각도 단위는 deg, SDK 호출 직전에만 rad 변환
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    from rustypot import Scs0009PyController
except ImportError as exc:
    raise ImportError(
        "rustypot 가 설치되어 있지 않습니다.\n"
        "다음 명령으로 설치하세요:\n"
        "    pip install rustypot"
    ) from exc

import config

LOG = logging.getLogger(__name__)


# =============================================================================
# 내부 유틸
# =============================================================================

# 손가락 이름 → 서보 ID 쌍
_FINGER_SERVOS: Dict[str, Tuple[int, int]] = {
    "index":  (1, 2),
    "middle": (3, 4),
    "ring":   (5, 6),
    "thumb":  (7, 8),
}


def _mirror_for_left_hand(pose: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """왼손(Side=2) 용 좌우 대칭 변환.

    Amazing_Hand_Demo.py 공식 예제의 Side==2 분기와 동일한 방식으로,
    두 서보의 각도 부호/순서를 바꿔 좌우 대칭을 만듭니다.
    """
    return {name: (-a2, -a1) for name, (a1, a2) in pose.items()}


# =============================================================================
# Controller
# =============================================================================

class AmazingHandController:
    """Amazing Hand 8-DOF 제어 (공식 rustypot SDK 사용)."""

    def __init__(
        self,
        port: str = config.HAND_PORT,
        baudrate: int = config.HAND_BAUDRATE,
        side: int = config.HAND_SIDE,
        event_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        port : str
            서보 버스 COM 포트 (config.HAND_PORT 기본값).
        baudrate : int
            Feetech SCS 표준 1 Mbps.
        side : int
            1 = Right Hand, 2 = Left Hand.
        event_callback : Optional[Callable[[str], None]]
            동작 phase 시작/종료 시 호출 — Teensy 동기화 마커 송신 용도.
            예) lambda name: sensor_reader.send_marker(name)
        """
        self.port = port
        self.baudrate = baudrate
        self.side = side
        self.event_callback = event_callback
        self._controller: Optional[Scs0009PyController] = None

    # -------------------------------------------------------------------------
    # 연결 / 해제
    # -------------------------------------------------------------------------

    def connect(self) -> None:
        """서보 버스 연결 + 토크 활성화."""
        LOG.info("Amazing Hand 연결: %s @ %d bps", self.port, self.baudrate)
        self._controller = Scs0009PyController(
            serial_port=self.port,
            baudrate=self.baudrate,
            timeout=0.5,
        )
        # 공식 예제와 동일: 1번 서보 토크 on 으로 전체 버스 활성화 트리거
        self._controller.write_torque_enable(1, 1)
        time.sleep(0.2)
        LOG.info("Amazing Hand 연결 완료 (Side=%d)", self.side)

    def disconnect(self) -> None:
        """연결 해제. 서보 토크는 유지 (손 자세 고정)."""
        self._controller = None
        LOG.info("Amazing Hand 연결 해제")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # -------------------------------------------------------------------------
    # 이벤트 마커
    # -------------------------------------------------------------------------

    def _mark(self, name: str) -> None:
        """Teensy 센서 로그와 동기화를 위한 이벤트 마커."""
        if self.event_callback is not None:
            try:
                self.event_callback(name)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("event_callback(%s) 실패: %s", name, exc)

    def _quiet_wait(self, action_name: str) -> None:
        """Action(tap/press/rub) 시작 전 서보 진동 감쇠 대기.

        목적
        ----
          자세 이동(hover 등) 직후에는 서보/구조체 잔류 진동이 남아 있어
          Knowles BU(16kHz) / WowSkin 신호를 오염시킨다.
          action 이 시작되기 전에 이 잔류 진동이 완전히 감쇠할 때까지 대기.

        동작
        ----
          pre_action_quiet_ms 만큼 무동작 대기.
          시작/종료 시점을 이벤트 마커로 기록하여 센서 CSV 와 동기화.
        """
        ms = config.SAFETY_LIMITS.pre_action_quiet_ms
        if ms <= 0:
            return
        self._mark(f"{action_name}_quiet_start")
        LOG.info("   [%s] 서보 진동 감쇠 대기 %d ms", action_name, ms)
        time.sleep(ms / 1000.0)
        self._mark(f"{action_name}_quiet_end")

    # -------------------------------------------------------------------------
    # 저수준: 단일 손가락 이동 (Amazing_Hand_Demo.py Move_* 함수 동일 패턴)
    # -------------------------------------------------------------------------

    def _move_finger(self, finger: str, angle_1: float, angle_2: float, speed: int) -> None:
        """공식 예제 Move_Index / Move_Middle / ... 와 동일 로직."""
        if self._controller is None:
            raise RuntimeError("connect() 먼저 호출하세요.")

        id1, id2 = _FINGER_SERVOS[finger]
        middle_1 = config.MIDDLE_POS[id1 - 1]
        middle_2 = config.MIDDLE_POS[id2 - 1]

        self._controller.write_goal_speed(id1, speed)
        time.sleep(0.0002)
        self._controller.write_goal_speed(id2, speed)
        time.sleep(0.0002)

        pos_1 = np.deg2rad(middle_1 + angle_1)
        pos_2 = np.deg2rad(middle_2 + angle_2)
        self._controller.write_goal_position(id1, pos_1)
        self._controller.write_goal_position(id2, pos_2)
        time.sleep(0.005)

    def _apply_pose(self, pose: Dict[str, Tuple[float, float]], speed: int) -> None:
        """네 손가락 동시 이동 (좌우 대칭 자동)."""
        if self.side == 2:
            pose = _mirror_for_left_hand(pose)
        for finger, (a1, a2) in pose.items():
            self._move_finger(finger, a1, a2, speed)

    # -------------------------------------------------------------------------
    # 고수준: phase 별 자세
    # -------------------------------------------------------------------------

    def home(self) -> None:
        """초기 자세 (모든 손가락 펼침)."""
        self._mark("home_start")
        self._apply_pose(config.POSE_HOME, config.SERVO_SPEEDS["home"])
        time.sleep(config.SAFETY_LIMITS.settle_ms / 1000.0)
        self._mark("home_done")

    def hover(self, hold_s: float = 0.0) -> None:
        """Hover 자세 (검지만 편 Index-Pointing).

        Parameters
        ----------
        hold_s : float
            자세 도달 후 추가로 유지할 시간(초). 0이면 기존 동작과 동일하게
            settle_ms 안정화만 하고 반환. >0 이면 그만큼 검지만 편 상태로 대기하며,
            hover_hold_start / hover_hold_end 마커로 센서 CSV 와 동기화.
        """
        self._mark("hover_start")
        self._apply_pose(config.POSE_HOVER, config.SERVO_SPEEDS["hover"])
        time.sleep(config.SAFETY_LIMITS.settle_ms / 1000.0)
        if hold_s > 0:
            self._mark("hover_hold_start")
            LOG.info("   [hover] 자세 유지 %.1f s", hold_s)
            time.sleep(hold_s)
            self._mark("hover_hold_end")
        self._mark("hover_done")

    def sniff(self) -> None:
        """Sniff 자세 (hover 와 동일, 가스 센서 안정화)."""
        self._mark("sniff_pose")
        self._apply_pose(config.POSE_SNIFF, config.SERVO_SPEEDS["sniff"])
        time.sleep(config.SAFETY_LIMITS.settle_ms / 1000.0)

    def safe(self) -> None:
        """실험 종료 안전 자세 (접힘)."""
        self._mark("safe_start")
        safe_pose = {name: (90.0, -90.0) for name in _FINGER_SERVOS}
        self._apply_pose(safe_pose, config.SERVO_SPEEDS["safe"])
        time.sleep(config.SAFETY_LIMITS.settle_ms / 1000.0)
        self._mark("safe_done")

    # -------------------------------------------------------------------------
    # 액션: tap / press / rub
    # -------------------------------------------------------------------------

    def tap(
        self,
        n_taps: Optional[int] = None,
        params: "config.TapParams" = config.TAP_PARAMS,
    ) -> int:
        """검지 tap 을 고정 횟수만큼 반복.

        한 tap = (contact → release) 한 쌍.

        Parameters
        ----------
        n_taps : Optional[int]
            반복 횟수. None 이면 params.repeats (기본 3) 사용.

        Returns
        -------
        int : 실행된 tap 횟수
        """
        n = n_taps if n_taps is not None else params.repeats
        self._mark("tap_phase_start")

        # --- 서보 진동 감쇠 대기 (센서 오염 방지) ---
        self._quiet_wait("tap")

        hover_ang = config.POSE_HOVER["index"]
        contact_ang = config.INDEX_CONTACT_ANGLE
        tap_speed = config.SERVO_SPEEDS["tap"]

        for i in range(n):
            self._mark(f"tap_{i+1}_contact")
            self._move_finger("index", *contact_ang, tap_speed)
            time.sleep(params.contact_ms / 1000.0)

            self._mark(f"tap_{i+1}_release")
            self._move_finger("index", *hover_ang, tap_speed)
            time.sleep(params.release_ms / 1000.0)

        self._mark("tap_phase_end")
        return n

    def press(self, params: "config.PressParams" = config.PRESS_PARAMS) -> None:
        """단일 긴 press: 천천히 내려가 → 유지 → 천천히 복귀.

        descend_ms + hold_ms + release_ms = PhaseTiming.press*1000 이 되어야 함.
        """
        self._mark("press_phase_start")

        # --- 서보 진동 감쇠 대기 ---
        self._quiet_wait("press")

        press_ang = config.INDEX_PRESS_ANGLE
        hover_ang = config.POSE_HOVER["index"]
        press_speed = config.SERVO_SPEEDS["press"]

        self._mark("press_descend")
        self._move_finger("index", *press_ang, press_speed)
        time.sleep(params.descend_ms / 1000.0)

        self._mark("press_hold")
        time.sleep(params.hold_ms / 1000.0)

        self._mark("press_release")
        self._move_finger("index", *hover_ang, press_speed)
        time.sleep(params.release_ms / 1000.0)

        self._mark("press_phase_end")

    def rub(
        self,
        cycles: int = config.RUB_CYCLES,
        amp_deg: float = config.RUB_AMP_DEG,
        params: "config.RubParams" = config.RUB_PARAMS,
    ) -> None:
        """검지 각도 진동 (Amazing Hand 제약상 각도 진동만 가능, 진짜 측방 문지름 아님).

        접촉 유지 각도(INDEX_CONTACT_ANGLE) 를 중심으로 ±amp_deg 왕복.
        """
        self._mark("rub_phase_start")

        # --- 서보 진동 감쇠 대기 ---
        self._quiet_wait("rub")

        base_a1, base_a2 = config.INDEX_CONTACT_ANGLE
        rub_speed = config.SERVO_SPEEDS["rub"]
        half_s = params.half_period_ms / 1000.0

        for i in range(cycles):
            # 진동 +쪽
            self._mark(f"rub_{i+1}_pos")
            self._move_finger(
                "index",
                base_a1 + amp_deg,
                base_a2 - amp_deg,
                rub_speed,
            )
            time.sleep(half_s)

            # 진동 -쪽
            self._mark(f"rub_{i+1}_neg")
            self._move_finger(
                "index",
                base_a1 - amp_deg,
                base_a2 + amp_deg,
                rub_speed,
            )
            time.sleep(half_s)

        # 진동 종료 후 접촉 유지 각도로 복귀
        self._move_finger("index", base_a1, base_a2, rub_speed)
        time.sleep(0.2)

        self._mark("rub_phase_end")