#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_register_dump.py — SCS0009 레지스터 전체 읽기
====================================================
서보 1번의 레지스터 0x00~0x50 전부를 1바이트씩 READ해서 출력한다.
현재 위치, 목표 위치 레지스터의 실제 주소를 역으로 확인한다.

① 이 스크립트 실행 → 전체 레지스터 값 확인
② 서보 축을 손으로 조금 돌린 뒤 재실행
③ 값이 바뀐 레지스터 = Present Position 주소

실행:
  python diag_register_dump.py
  python diag_register_dump.py --port COM4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))

import serial
from config import HAND_PORT, HAND_BAUD

INST_PING  = 0x01
INST_READ  = 0x02
INST_WRITE = 0x03


def checksum(data: list[int]) -> int:
    return (~sum(data)) & 0xFF


def send_packet(ser: serial.Serial, servo_id: int, inst: int, params: list[int]) -> None:
    length = len(params) + 2
    body   = [servo_id, length, inst] + params
    cs     = checksum(body)
    pkt    = bytes([0xFF, 0xFF] + body + [cs])
    ser.write(pkt)


def read_bytes(ser: serial.Serial, n: int) -> bytes:
    """블로킹 read, timeout 후 받은 것만 반환."""
    return ser.read(n)


def ping(ser: serial.Serial, servo_id: int) -> bool:
    ser.reset_input_buffer()
    send_packet(ser, servo_id, INST_PING, [])
    raw = read_bytes(ser, 6)
    return len(raw) >= 4 and raw[0] == 0xFF and raw[1] == 0xFF and raw[2] == servo_id


def read_register(ser: serial.Serial, servo_id: int, reg: int, n_bytes: int = 1) -> list[int]:
    """
    레지스터 reg 에서 n_bytes 읽기.
    응답 패킷: FF FF ID LEN ERR DATA... CS
    실패 시 빈 리스트 반환.
    """
    ser.reset_input_buffer()
    send_packet(ser, servo_id, INST_READ, [reg, n_bytes])

    total = 6 + n_bytes  # header(4) + err(1) + data(n) + cs(1)
    raw = read_bytes(ser, total)

    if len(raw) < 4:
        return []
    if raw[0] != 0xFF or raw[1] != 0xFF or raw[2] != servo_id:
        return []
    length = raw[3]
    if len(raw) < 4 + length:
        return []
    # error = raw[4]
    data = list(raw[5 : 4 + length - 1])   # -1 for checksum
    return data


def dump_registers(ser: serial.Serial, servo_id: int, start: int = 0x00, end: int = 0x50) -> dict:
    result = {}
    for reg in range(start, end + 1):
        data = read_register(ser, servo_id, reg, 1)
        val = data[0] if data else None
        result[reg] = val
        time.sleep(0.003)
    return result


def print_dump(dump: dict, label: str = "") -> None:
    if label:
        print(f"\n{'━'*54}")
        print(f"  {label}")
        print(f"{'━'*54}")

    # 알려진 레지스터 이름
    names = {
        0x05: "ID",
        0x06: "Baud Rate",
        0x08: "Return Level",
        0x28: "Torque Enable",
        0x29: "LED",
        0x2A: "Goal Pos L",
        0x2B: "Goal Pos H",
        0x2C: "Goal Time/Spd L",
        0x2D: "Goal Time/Spd H",
        0x2E: "Goal Spd/Time L",
        0x2F: "Goal Spd/Time H",
        0x37: "Lock",
        0x38: "Present Pos L",
        0x39: "Present Pos H",
        0x3A: "Present Spd L",
        0x3B: "Present Spd H",
        0x42: "Moving",
    }

    print(f"\n  {'REG':>5}  {'HEX':>5}  {'DEC':>5}  {'BIN':>10}  NAME")
    print(f"  {'---':>5}  {'---':>5}  {'---':>5}  {'---':>10}  ----")
    for reg, val in sorted(dump.items()):
        if val is None:
            s = f"  0x{reg:02X} ({reg:3d})  {'?':>5}  {'?':>5}  {'?':>10}"
        else:
            s = (f"  0x{reg:02X} ({reg:3d})"
                 f"  0x{val:02X}  {val:5d}  {val:08b}")
        name = names.get(reg, "")
        print(s + (f"  ← {name}" if name else ""))

    # 2바이트 위치 값 계산
    print()
    if dump.get(0x2A) is not None and dump.get(0x2B) is not None:
        goal = dump[0x2A] | (dump[0x2B] << 8)
        print(f"  Goal Pos (0x2A+2B)    = {goal}  (±35°→401, 90°→827 for ID=1)")
    if dump.get(0x38) is not None and dump.get(0x39) is not None:
        pres = dump[0x38] | (dump[0x39] << 8)
        print(f"  Present Pos (0x38+39) = {pres}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=HAND_PORT)
    parser.add_argument("--id",   type=int, default=1)
    args = parser.parse_args()

    print(f"\n서보 ID={args.id}, 포트={args.port}, 보드레이트={HAND_BAUD}")

    with serial.Serial(args.port, HAND_BAUD, timeout=0.15, write_timeout=0.10) as ser:
        time.sleep(0.2)

        # PING 확인
        if not ping(ser, args.id):
            print(f"  PING 실패 (ID={args.id}). 포트/전원 확인.")
            sys.exit(1)
        print(f"  PING OK (ID={args.id})\n")

        # Return Level 확인
        rl = read_register(ser, args.id, 0x08, 1)
        if rl:
            print(f"  Return Level (0x08) = {rl[0]}")
            if rl[0] == 0:
                print("  ⚠  Return Level=0: 서보가 PING 외 명령에 응답 안 함")
                print("     READ 명령이 모두 실패할 수 있음 — 값이 '?' 로 표시됨")
            elif rl[0] == 1:
                print("  Return Level=1: READ 에만 응답")
            else:
                print("  Return Level=2: 모든 명령에 응답")
        print()

        # 전체 덤프
        print("레지스터 덤프 중... (약 3초)")
        dump = dump_registers(ser, args.id, start=0x00, end=0x50)
        print_dump(dump, f"레지스터 덤프 — ID={args.id}")

        # 토크 ON 후 비교
        input("\n\n  [Enter] 토크 ON 후 손가락 살짝 움직여보기 → 다시 덤프")

        # 토크 ON
        send_packet(ser, args.id, INST_WRITE, [0x28, 1])
        time.sleep(0.05)
        ser.reset_input_buffer()

        print("토크 ON 후 레지스터 덤프 중...")
        dump2 = dump_registers(ser, args.id, start=0x00, end=0x50)
        print_dump(dump2, "토크 ON 후 덤프")

        # 차이 출력
        changed = {r: (dump.get(r), dump2.get(r))
                   for r in set(dump) | set(dump2)
                   if dump.get(r) != dump2.get(r)}
        if changed:
            print("\n  [변경된 레지스터]")
            for reg, (before, after) in sorted(changed.items()):
                print(f"    0x{reg:02X} ({reg:3d}): {before} → {after}")
        else:
            print("\n  변경된 레지스터 없음 (모든 READ 실패 또는 값 동일)")

        # 토크 OFF
        send_packet(ser, args.id, INST_WRITE, [0x28, 0])
        time.sleep(0.05)


if __name__ == "__main__":
    main()
