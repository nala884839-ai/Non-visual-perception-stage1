#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hand_test.py — Amazing Hand 동작 검증 (3-3 단계)
================================================
hand_controller.AmazingHandController 로 손을 단계별로 움직여
서보 통신 / 전원 / 동작을 육안 확인한다.

실행:
  python hand_test.py                 # 대화형: 각 동작을 Enter로 하나씩
  python hand_test.py --auto          # 자동: 모든 동작을 순서대로
  python hand_test.py --port COM4     # 포트 지정 (기본 config.HAND_PORT)

각 동작 후 손을 눈으로 확인:
  · home  : 기본 자세 (모든 손가락 중립)
  · hover : 손 펴기 (물체 접근 준비 자세)
  · tap   : 검지로 톡 (action)
  · press : 검지로 눌러 유지
  · rub   : 검지로 문지르기
  · safe  : 안전 자세 (마무리)
"""

import argparse
import sys
import time

# event_callback: 마커를 콘솔에 찍어 동작 타이밍 확인
def mark(name: str) -> None:
    print(f"    [marker] {name}")


def pause(auto: bool, msg: str) -> None:
    if auto:
        time.sleep(1.2)
    else:
        input(f"    >>> Enter 를 누르면 [{msg}] 실행 (Ctrl+C 로 중단)... ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="시리얼 포트 (기본 config.HAND_PORT)")
    ap.add_argument("--auto", action="store_true", help="자동 진행 (Enter 없이)")
    args = ap.parse_args()

    import config
    from hand_controller import AmazingHandController

    port = args.port or config.HAND_PORT
    print(f"[hand_test] 포트 {port} 로 Amazing Hand 연결 시도")
    print(f"[hand_test] side={config.HAND_SIDE}  baud={config.HAND_BAUDRATE}")
    print("-" * 56)

    hand = AmazingHandController(
        port=port,
        baudrate=config.HAND_BAUDRATE,
        side=config.HAND_SIDE,
        event_callback=mark,
    )

    try:
        hand.connect()
        print("[hand_test] 연결 성공 ✓\n")
    except Exception as exc:
        print(f"[hand_test] 연결 실패: {exc}")
        print("  - 포트 번호 확인 (장치관리자), 핸드 5V 전원 확인, rustypot 설치 확인")
        return 1

    # 동작 시퀀스: (이름, 호출 함수)
    steps = [
        ("home",  lambda: hand.home()),
        ("hover", lambda: hand.hover()),
        ("tap",   lambda: hand.tap()),
        ("press", lambda: hand.press()),
        ("rub",   lambda: hand.rub()),
        ("home",  lambda: hand.home()),
        ("safe",  lambda: hand.safe()),
    ]

    try:
        for name, fn in steps:
            pause(args.auto, name)
            print(f"  → {name} 실행")
            fn()
            print(f"    {name} 완료 (손 동작 육안 확인)\n")
            time.sleep(0.3)
        print("[hand_test] 전체 동작 완료 ✓")
        print("  서보 8개가 떨림/이상음 없이 부드럽게 움직였으면 정상.")
    except KeyboardInterrupt:
        print("\n[hand_test] 사용자 중단 — 안전 자세로 복귀")
        try:
            hand.safe()
        except Exception:
            pass
    except Exception as exc:
        print(f"\n[hand_test] 동작 중 오류: {exc}")
        return 1
    finally:
        hand.disconnect()
        print("[hand_test] 연결 해제")

    return 0


if __name__ == "__main__":
    sys.exit(main())
