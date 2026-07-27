#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_servo.py — SCS0009 서보 통신 진단 스크립트
=================================================
손이 안 움직일 때 실행. 세 가지 방식으로 서보 1번을 움직여본다.
  Step A: REG_WRITE (0x04) + ACTION (0x05)  — Speed/Time 순서 수정 버전
  Step B: WRITE (0x03) 직접 쓰기
  Step C: Position-only WRITE (Speed/Time 없이 위치만)

각 Step 후 Enter → 서보가 실제로 움직였는지 육안 확인.

실행:
  python diag_servo.py
  python diag_servo.py --port COM4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))

import serial

from config import HAND_PORT, HAND_BAUD, MIDDLE_POS, STEP_DEG

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diag")

# ── 레지스터 주소 ─────────────────────────────────────────────────────────
REG_TORQUE_ENABLE      = 0x28
REG_GOAL_POSITION_L    = 0x2A
REG_GOAL_SPEED_L       = 0x2C   # SCS 순서: Speed 먼저
REG_GOAL_TIME_L        = 0x2E
REG_PRESENT_POSITION_L = 0x38

INST_PING      = 0x01
INST_READ      = 0x02
INST_WRITE     = 0x03
INST_REG_WRITE = 0x04
INST_ACTION    = 0x05
BROADCAST_ID   = 0xFE


def checksum(data: list[int]) -> int:
    return (~sum(data)) & 0xFF


def send_packet(ser: serial.Serial, servo_id: int, instruction: int, params: list[int]) -> bytes:
    length = len(params) + 2
    body   = [servo_id, length, instruction] + params
    cs     = checksum(body)
    pkt    = bytes([0xFF, 0xFF] + body + [cs])
    log.debug(f"  TX [{servo_id}] {pkt.hex(' ')}")
    ser.write(pkt)
    return pkt


def read_response(ser: serial.Serial, expected_id: int, n_data: int = 0) -> bytes:
    """응답 패킷 수신 — 원시 bytes 반환."""
    total = 6 + n_data   # FF FF ID LEN ERR [DATA...] CS
    raw = ser.read(total)
    log.debug(f"  RX [{expected_id}] {raw.hex(' ') if raw else '(timeout)'}")
    return raw


def ping_servo(ser: serial.Serial, servo_id: int) -> bool:
    ser.reset_input_buffer()
    send_packet(ser, servo_id, INST_PING, [])
    raw = read_response(ser, servo_id, 0)
    if len(raw) >= 4 and raw[0] == 0xFF and raw[1] == 0xFF and raw[2] == servo_id:
        log.info(f"  → PING OK (ID={servo_id})")
        return True
    log.warning(f"  → PING 실패 (ID={servo_id}), raw={raw.hex()}")
    return False


def torque_enable(ser: serial.Serial, servo_id: int, enable: bool) -> None:
    val = 1 if enable else 0
    send_packet(ser, servo_id, INST_WRITE, [REG_TORQUE_ENABLE, val])
    time.sleep(0.010)   # 응답 대기 (읽지 않고 flush 용)
    ser.reset_input_buffer()


def deg_to_pos(servo_id: int, deg: float) -> int:
    return round(MIDDLE_POS[servo_id - 1] + deg / STEP_DEG)


def u16le(val: int) -> tuple[int, int]:
    """little-endian (L, H) 변환 — SCS0009 전용.
    SCS0009 레지스터 맵: 낮은 주소 = LSB, Feetech SDK scs_makeword 기준."""
    v = val & 0xFFFF
    return v & 0xFF, (v >> 8) & 0xFF


# ── 이동 방식 A: REG_WRITE + ACTION (little-endian) ──────────────────────
def move_A(ser: serial.Serial, servo_id: int, pos: int, speed: int) -> None:
    pL, pH = u16le(pos)
    sL, sH = u16le(speed)
    params = [
        REG_GOAL_POSITION_L,
        pL, pH,   # Position L/H (little-endian)
        0,  0,    # Time = 0     (속도 제어)
        sL, sH,   # Speed L/H   (little-endian)
    ]
    send_packet(ser, servo_id, INST_REG_WRITE, params)
    time.sleep(0.003)
    body = [BROADCAST_ID, 2, INST_ACTION]
    cs   = checksum(body)
    pkt  = bytes([0xFF, 0xFF] + body + [cs])
    log.debug(f"  ACTION {pkt.hex(' ')}")
    ser.write(pkt)


# ── 이동 방식 B: WRITE 직접 (little-endian) ──────────────────────────────
def move_B(ser: serial.Serial, servo_id: int, pos: int, speed: int) -> None:
    pL, pH = u16le(pos)
    sL, sH = u16le(speed)
    params = [
        REG_GOAL_POSITION_L,
        pL, pH,
        0,  0,
        sL, sH,
    ]
    send_packet(ser, servo_id, INST_WRITE, params)


# ── 이동 방식 C: WRITE (위치만, little-endian) ───────────────────────────
def move_C(ser: serial.Serial, servo_id: int, pos: int) -> None:
    pL, pH = u16le(pos)
    params = [REG_GOAL_POSITION_L, pL, pH]
    send_packet(ser, servo_id, INST_WRITE, params)


def pause(msg: str) -> None:
    input(f"\n  ▷ {msg}  [Enter]\n")


def run(port: str) -> None:
    SERVO_ID = 1
    pos_open  = deg_to_pos(SERVO_ID, -35.0)   # ≈ 401
    pos_close = deg_to_pos(SERVO_ID,  90.0)   # ≈ 827

    log.info(f"포트 {port} 열기...")
    with serial.Serial(port, HAND_BAUD, timeout=0.20, write_timeout=0.10) as ser:
        time.sleep(0.2)

        print(f"\n{'='*54}")
        print(f"  SCS0009 진단 — 서보 ID={SERVO_ID}")
        print(f"  pos_open={pos_open}, pos_close={pos_close}")
        print(f"{'='*54}\n")

        # ── PING ──────────────────────────────────────────────────────────
        print("[PING] 서보 응답 확인")
        if not ping_servo(ser, SERVO_ID):
            print("  PING 실패 → COM 포트/전원 확인 후 재시도")
            return

        # ── 토크 활성화 ───────────────────────────────────────────────────
        print("\n[TORQUE] ON")
        torque_enable(ser, SERVO_ID, True)
        pause("토크 ON 완료. 서보 축을 손으로 돌려보세요 — 저항이 느껴지면 토크 정상.")

        # ── Step A: REG_WRITE + ACTION ────────────────────────────────────
        print("\n[STEP A] REG_WRITE(0x04) + ACTION(0x05) — pos_close")
        move_A(ser, SERVO_ID, pos_close, 1200)
        time.sleep(2.0)
        pause("서보가 움직였나요? (Y/N 기억해두기)")

        print("[STEP A] REG_WRITE — pos_open")
        move_A(ser, SERVO_ID, pos_open, 1200)
        time.sleep(2.0)
        pause("서보가 원위치로 돌아왔나요?")

        # ── Step B: WRITE 직접 ────────────────────────────────────────────
        print("\n[STEP B] WRITE(0x03) 직접 — pos_close")
        move_B(ser, SERVO_ID, pos_close, 1200)
        time.sleep(2.0)
        pause("서보가 움직였나요?")

        print("[STEP B] WRITE — pos_open")
        move_B(ser, SERVO_ID, pos_open, 1200)
        time.sleep(2.0)
        pause("원위치 복귀?")

        # ── Step C: 위치만 WRITE ──────────────────────────────────────────
        print("\n[STEP C] WRITE(위치 2바이트만) — pos_close")
        move_C(ser, SERVO_ID, pos_close)
        time.sleep(2.0)
        pause("서보가 움직였나요?")

        print("[STEP C] WRITE — pos_open")
        move_C(ser, SERVO_ID, pos_open)
        time.sleep(2.0)
        pause("원위치 복귀?")

        # ── 토크 해제 ─────────────────────────────────────────────────────
        print("\n[TORQUE] OFF")
        torque_enable(ser, SERVO_ID, False)

    print("\n진단 완료. 어떤 Step 에서 움직였는지 알려주세요.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=HAND_PORT)
    args = parser.parse_args()
    run(args.port)


if __name__ == "__main__":
    main()
