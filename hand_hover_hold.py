#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hand_hover_hold.py — 손을 hover(다 편) 자세로 만들고 그대로 유지
================================================================
hover 자세를 잡은 뒤, Ctrl+C 를 누를 때까지 그 자세를 계속 유지한다.
(SCS0009 서보는 토크가 걸린 채 전원이 유지되면 위치를 고정한다)

실행:
  python hand_hover_hold.py                 # 무한 유지, Ctrl+C 로 종료
  python hand_hover_hold.py --port COM5     # 포트 지정
  python hand_hover_hold.py --on-exit safe  # 종료 시 손을 안전 자세로 접음
  python hand_hover_hold.py --on-exit hold  # 종료 후에도 토크 유지(기본)

종료 옵션(--on-exit):
  hold (기본) : 프로그램 종료 후에도 서보 토크 유지 → 손이 hover 자세 그대로
  safe        : 종료 시 safe()로 손을 접고 종료
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
    print(f"[hover-hold] {port} 연결 시도 (side={config.HAND_SIDE})")

    hand = AmazingHandController(
        port=port,
        baudrate=config.HAND_BAUDRATE,
        side=config.HAND_SIDE,
    )

    try:
        hand.connect()
        print("[hover-hold] 연결 성공 → hover 실행")
        hand.hover()
        print("[hover-hold] 손 펴기 완료 — 자세 유지 중")
        print("[hover-hold] 종료하려면 Ctrl+C 를 누르세요")

        # 자세 유지: 연결을 유지한 채 대기 (서보 토크가 위치 고정)
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[hover-hold] 종료 요청 감지")
        if args.on_exit == "safe":
            try:
                print("[hover-hold] safe 자세로 접는 중...")
                hand.safe()
            except Exception as exc:
                print(f"[hover-hold] safe 중 오류: {exc}")
        else:
            print("[hover-hold] 토크 유지 — 손은 hover 자세 그대로 둠")
    except Exception as exc:
        print(f"[hover-hold] 오류: {exc}")
        return 1
    finally:
        hand.disconnect()   # 토크는 유지됨(자세 고정)
        print("[hover-hold] 연결 해제 (종료)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
