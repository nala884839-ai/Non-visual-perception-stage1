#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hand_hover.py — 손을 hover(다 편) 상태로 만든다
================================================
실행:
  python hand_hover.py                # config.HAND_PORT 사용
  python hand_hover.py --port COM5    # 포트 지정
  python hand_hover.py --hold 5       # hover 후 5초 유지하고 종료

기본 동작: connect → hover → (유지) → disconnect
"""

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="시리얼 포트 (기본 config.HAND_PORT)")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="hover 후 유지 시간(초). 0이면 바로 종료")
    args = ap.parse_args()

    import config
    from hand_controller import AmazingHandController

    port = args.port or config.HAND_PORT
    print(f"[hover] {port} 연결 시도 (side={config.HAND_SIDE})")

    hand = AmazingHandController(
        port=port,
        baudrate=config.HAND_BAUDRATE,
        side=config.HAND_SIDE,
    )

    try:
        hand.connect()
        print("[hover] 연결 성공 → hover 실행")
        hand.hover()
        print("[hover] 손 펴기 완료")
        if args.hold > 0:
            print(f"[hover] {args.hold}초 유지...")
            time.sleep(args.hold)
    except Exception as exc:
        print(f"[hover] 오류: {exc}")
        return 1
    finally:
        hand.disconnect()
        print("[hover] 연결 해제")
    return 0


if __name__ == "__main__":
    sys.exit(main())
