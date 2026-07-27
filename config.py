"""
config.py
비시각적 멀티모달 물체 인식 시스템 — 중앙 설정
==================================================

Amazing Hand + Teensy 센서 통합 실험의 모든 파라미터를 한 곳에 모읍니다.
하드웨어 교체 / 캘리브레이션 시 이 파일만 수정하면 됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# =============================================================================
# 1) 시리얼 포트 설정
# =============================================================================

# ※ 실제 COM 번호는 장치관리자/Arduino IDE 에서 확인 후 맞출 것.
#   (Hand=서보 USB-TTL, Teensy=센서 USB — 서로 바뀌지 않게 주의)
HAND_PORT: str = "COM4"          # Amazing Hand 서보 버스 (rustypot)
HAND_BAUDRATE: int = 1_000_000   # Feetech SCS0009 표준 1Mbps

TEENSY_PORT: str = "COM3"        # Teensy 4.1 USB Serial
TEENSY_BAUDRATE: int = 2_000_000 # teensy_sensor_logger.ino 와 동일

STAGE_PORT: str = "COM10"         # Arduino Uno (리니어 스테이지)
STAGE_BAUDRATE: int = 9_600      # LinearStage_Commanded.ino 와 동일


# =============================================================================
# 2) Amazing Hand 캘리브레이션
# =============================================================================
#   - Side: 1 = Right Hand, 2 = Left Hand
#   - MIDDLE_POS: 8개 서보의 기계적 중립 위치 (deg 단위)
#     ※ AmazingHand_Hand_FingerMiddlePos.py 로 각 서보 캘리브레이션 후 값 대입
#   - 서보 ID 매핑:  1,2=Index / 3,4=Middle / 5,6=Ring / 7,8=Thumb

HAND_SIDE: int = 1

# (deg 단위, Amazing_Hand_Demo.py MiddlePos 와 동일 의미)
MIDDLE_POS: List[float] = [3, 0, -5, -8, -2, 5, -12, 0]


# =============================================================================
# 3) 서보 속도 (rustypot write_goal_speed 단위: 0~7)
# =============================================================================
#   7 = MaxSpeed, 낮을수록 느림
#   Tap 은 빠르게, Press 는 천천히, Rub 은 중간 속도로

SERVO_SPEEDS: Dict[str, int] = {
    "home":   7,   # 초기 자세 이동 (빠르게)
    "hover":  5,   # 물체 위로 이동
    "sniff":  4,   # 가스 센서 흡입 자세 (조용히)
    "tap":    7,   # Tap: 최대 속도
    "press":  2,   # Press: 느린 하강
    "rub":    5,   # Rub: 중간 속도 진동
    "safe":   6,   # 실험 종료 안전 복귀
}


# =============================================================================
# 4) 자세별 손가락 각도 (deg, MIDDLE_POS 기준 상대값)
# =============================================================================
#   각 자세는 (finger_name → (servo_angle_1, servo_angle_2)) 형태
#
#   설계 의도:
#     - 검지(Index) = 주 접촉 손가락: SingleTact 촉각(압력) + Knowles BU 진동
#     - 중지/약지/엄지 = 접힘(curled) → 물체와 간섭 방지
#     - 손바닥(palm) 은 물체를 향하게 마운트 → 가스 센서 근접
#python hand_tap_hold.py --angle 55
#   Hover: 물체 위 ~10mm, 검지 살짝 편 상태 (접촉 직전)
#   Sniff: Hover 와 같은 손 모양, 가스 센서가 물체 냄새 맡는 구간
#   Tap/Press/Rub: 검지만 움직여서 접촉/진동 생성

# 오른손 (Side=1) 기준 각도. 왼손은 트리거 시 자동 대칭 변환.

POSE_HOME: Dict[str, Tuple[float, float]] = {
    # 모든 손가락 펼침 (OpenHand)
    "index":  (-35.0,  35.0),
    "middle": (-35.0,  35.0),
    "ring":   (-35.0,  35.0),
    "thumb":  (-35.0,  35.0),
}

POSE_HOVER: Dict[str, Tuple[float, float]] = {
    # 네 손가락 모두 편 상태 유지 (다른 손가락 안 움직임 → 배선 꼬임 방지)
    # 검지만 tap/press/rub 로 굽혀 접촉
    "index":  (-35.0,  35.0),
    "middle": (-35.0,  35.0),
    "ring":   (-35.0,  35.0),
    "thumb":  (-35.0,  35.0),
}

POSE_SNIFF: Dict[str, Tuple[float, float]] = {
    # Hover 와 동일 — 네 손가락 모두 편 상태 유지 (가스 확산 안정화)
    "index":  (-35.0,  35.0),
    "middle": (-35.0,  35.0),
    "ring":   (-35.0,  35.0),
    "thumb":  (-35.0,  35.0),
}

# -----------------------------------------------------------------------------
# 검지(Index) 동작 각도 — tap / press / rub 의 핵심 파라미터
# -----------------------------------------------------------------------------

# Hover 에서 '접촉 완료' 위치까지 검지가 굽는 각도
# (tap 을 press 깊이까지 깊게: 20 → 35)
INDEX_CONTACT_ANGLE: Tuple[float, float] = (50.0, -50.0)
# Press 시 검지가 더 깊게 굽는 각도 (더 큰 힘)
INDEX_PRESS_ANGLE:   Tuple[float, float] = (45.0, -45.0)

# Rub 진동 진폭 (접촉 유지 각도 ± 이 값만큼 좌우로 흔듦)
RUB_AMP_DEG: float = 10.0
# Rub 사이클 수 (FFT/주파수 분석에 충분한 최소 반복, 모터 소음 최소화)
RUB_CYCLES: int = 5


# =============================================================================
# 5) 1-Trial 타이밍 (초 단위)
# =============================================================================
#   세 가지 Trial 변형에서 공통으로 사용
#   총 길이는 variant 별로 다름:
#     variant_tap        : 5 + 10 + 15 + 10 + 15 = 55s
#     variant_tap_press  : 5 + 10 + 15 + 10 + 10 + 15 = 65s
#     variant_all        : 5 + 10 + 15 + 10 + 10 + 10 + 15 = 75s

@dataclass
class PhaseTiming:
    baseline: float = 5.0    # 정적 baseline (센서 안정화)
    move:     float = 2.0    # 스테이지 전진 후 안정화 대기 (실제 이동은 Uno blocking)
    back:     float = 2.0    # 스테이지 복귀 후 대기
    hover:    float = 10.0   # 물체 위 호버링
    sniff:    float = 25.0   # 가스 흡입 (15→25: 가스 곡선 피처용 점 확보 + BME688 2사이클+)
    tap:      float = 10.0   # Tap phase 길이 (실제 tap 횟수는 TapParams.repeats=3)
    press:    float = 10.0   # Press + hold (single sustained press)
    rub:      float = 10.0   # Rub 진동 (RUB_CYCLES 만큼)
    recovery: float = 15.0   # 손 복귀 + 센서 회복


# =============================================================================
# 6) Tap / Rub 내부 파라미터
# =============================================================================

@dataclass
class TapParams:
    """Tap phase 내에서 반복되는 개별 tap 의 파라미터."""
    repeats:    int = 3      # tap 반복 횟수 (모터 소음 최소화)
    contact_ms: int = 250    # 접촉 유지 시간 (150→250: steady-state 압력 신호 확보)
    release_ms: int = 500    # tap 사이 간격 (300→500: 잔류 진동·force 완전 회복)
    # 전체 tap 소요 = repeats × (contact_ms + release_ms)
    #               = 3 × 750 = 2,250 ms
    # phase 나머지 시간은 센서(특히 Knowles BU) 진동 감쇠용 hover 유지


@dataclass
class PressParams:
    """Press phase 파라미터 (단일 긴 press)."""
    descend_ms: int = 2_000  # 천천히 내려가는 시간
    hold_ms:    int = 5_000  # 접촉 유지 (촉각/가스 정상상태 신호 포착에 충분)
    release_ms: int = 2_000  # 복귀
    # descend + hold + release = 9,000ms
    # + pre_action_quiet (1s) = 10,000ms = PhaseTiming.press


@dataclass
class RubParams:
    """Rub phase 파라미터."""
    half_period_ms: int = 500  # 한쪽 방향 이동 시간 → 전체 주기 = 1s
    # 전체 사이클 수 = RUB_CYCLES (config 상단)


# =============================================================================
# 7) 안전 한계
# =============================================================================

@dataclass
class SafetyLimits:
    max_trial_seconds:   float = 120.0   # 1 trial 최대 허용 시간 (hang 방지)
    max_servo_load:      int   = 900     # SCS0009 load 경고 한계 (0~1023)
    settle_ms:           int   = 200     # 자세 전환 후 짧은 안정화 (포즈 도달 확인용)
    pre_action_quiet_ms: int   = 1000    # action(tap/press/rub) 시작 전 무음 대기
                                         # ↳ 서보 위치 도달 후 진동이 완전히 감쇠하여
                                         #   센서(특히 Knowles BU 16kHz) 오염이 최소화될 때까지 대기


# =============================================================================
# 8) 데이터 저장 경로 템플릿
# =============================================================================

DATA_ROOT: str = "data/raw"
TRIAL_DIR_TEMPLATE: str = "{session}/obj{obj_id:02d}_{obj_name}/trial_{trial_id:03d}_{variant}"


# =============================================================================
# 9) Trial 변형 정의 (phase 순서)
# =============================================================================

TRIAL_VARIANTS: Dict[str, List[str]] = {
    "tap":            ["baseline", "move", "hover", "sniff", "tap",                 "hover", "back", "recovery"],
    "tap_press":      ["baseline", "move", "hover", "sniff", "tap", "press",        "hover", "back", "recovery"],
    "tap_press_rub":  ["baseline", "move", "hover", "sniff", "tap", "press", "rub", "hover", "back", "recovery"],
}


# =============================================================================
# 10) 싱글톤 인스턴스 (import 시 바로 사용 가능)
# =============================================================================

PHASE_TIMING       = PhaseTiming()
TAP_PARAMS         = TapParams()
PRESS_PARAMS       = PressParams()
RUB_PARAMS         = RubParams()
SAFETY_LIMITS      = SafetyLimits()
