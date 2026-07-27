#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hand_open_hold.py — 손가락을 모두 편 자세(home/OpenHand)로 유지
================================================================
모든 손가락을 편 'home' 자세를 잡고, Ctrl+C 까지 그 자세를 유지한다.

  · home  = 네 손가락 모두 펼침 (OpenHand)   ← 이 스크립트
  · hover = 검지만 펴고 나머지는 접힘 (포인팅)  ← 다른 스크립트

실행:
  python hand_open_hold.py                 # 무한 유지, Ctrl+C 로 종료
  python hand_open_hold.py --port COM5     # 포트 지정
  python hand_open_hold.py --on-exit safe  # 종료 시 손 접기
  python hand_open_hold.py --on-exit hold  # 종료 후에도 자세 유지(기본)
"""

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="시리얼 포트 (기본 config.HAND_PORT)")
    ap.add_argument("--on-exit", choices=["hold", "safe"], default="hold",
                    help="종료 시 동작: hold(자세 유지) / safe(접기)")
    args = ap.parse_args()

    import config
    from hand_controller import AmazingHandController

    port = args.port or config.HAND_PORT
    print(f"[open-hold] {port} 연결 시도 (side={config.HAND_SIDE})")

    hand = AmazingHandController(
        port=port,
        baudrate=config.HAND_BAUDRATE,
        side=config.HAND_SIDE,
    )

    try:
        hand.connect()
        print("[open-hold] 연결 성공 → home(모든 손가락 펼침) 실행")
        hand.home()
        print("[open-hold] 손가락 모두 편 자세 완료 — 유지 중")
        print("[open-hold] 종료하려면 Ctrl+C 를 누르세요")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[open-hold] 종료 요청 감지")
        if args.on_exit == "safe":
            try:
                print("[open-hold] safe 자세로 접는 중...")
                hand.safe()
            except Exception as exc:
                print(f"[open-hold] safe 중 오류: {exc}")
        else:
            print("[open-hold] 토크 유지 — 손은 편 자세 그대로 둠")
    except Exception as exc:
        print(f"[open-hold] 오류: {exc}")
        return 1
    finally:
        hand.disconnect()
        print("[open-hold] 연결 해제 (종료)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
