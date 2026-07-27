#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hand_tap_hold.py — tap 최종 접촉 포즈로 고정 (접촉면/패드 위치 확인용)
====================================================================
tap 동작의 '접촉 완료' 순간 자세를 그대로 만들어 유지한다.
검지가 INDEX_CONTACT_ANGLE(현재 50도)로 굽은 상태에서 멈춰,
물체와 검지의 접촉면 / SingleTact 패드 위치를 눈으로 천천히 확인 가능.

  · 다른 손가락(중지·약지·엄지): home 자세(다 폄) 유지 — tap 과 동일
  · 검지: 접촉 각도로 굽힌 채 고정

실행:
  python hand_tap_hold.py                 # 검지 접촉 포즈로 고정, Ctrl+C 종료
  python hand_tap_hold.py --port COM5
  python hand_tap_hold.py --angle 55      # 접촉 각도 직접 지정(테스트용)
  python hand_tap_hold.py --on-exit safe  # 종료 시 손 접기

※ 실제 tap 과 동일하게, 물체를 스테이지 앞 제 위치에 놓고 확인하면
  tap 시 검지가 물체의 어디에 닿는지 정확히 볼 수 있다.
※ 서보가 과부하(떨림/소음)면 --angle 로 각도를 낮춰 확인.
"""

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="시리얼 포트 (기본 config.HAND_PORT)")
    ap.add_argument("--angle", type=float, default=None,
                    help="검지 접촉 각도(deg). 기본 config.INDEX_CONTACT_ANGLE")
    ap.add_argument("--on-exit", choices=["hold", "safe"], default="hold",
                    help="종료 시: hold(자세 유지) / safe(접기)")
    args = ap.parse_args()

    import config
    from hand_controller import AmazingHandController

    port = args.port or config.HAND_PORT

    # 접촉 각도 결정 (기본은 config 값, --angle 로 덮어쓰기 가능)
    if args.angle is not None:
        contact_ang = (args.angle, -args.angle)
    else:
        contact_ang = config.INDEX_CONTACT_ANGLE

    print(f"[tap-hold] {port} 연결 (side={config.HAND_SIDE})")
    print(f"[tap-hold] 검지 접촉 각도 = {contact_ang}")

    hand = AmazingHandController(
        port=port,
        baudrate=config.HAND_BAUDRATE,
        side=config.HAND_SIDE,
    )

    try:
        hand.connect()
        print("[tap-hold] 연결 성공")

        # 1) home: 네 손가락 모두 폄 (tap 시작 자세와 동일)
        print("[tap-hold] home 자세 (모든 손가락 폄)")
        hand.home()
        time.sleep(0.5)

        # 2) 검지만 접촉 각도로 굽힘 (tap 의 '접촉 완료' 순간)
        print("[tap-hold] 검지를 접촉 포즈로 굽힘 → 고정")
        hand._move_finger("index", *contact_ang, config.SERVO_SPEEDS["tap"])

        print("[tap-hold] === tap 최종 접촉 포즈 고정됨 ===")
        print("[tap-hold] 검지 접촉면 / SingleTact 패드 위치를 확인하세요")
        print("[tap-hold] 종료: Ctrl+C")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[tap-hold] 종료 요청")
        if args.on_exit == "safe":
            try:
                print("[tap-hold] safe 자세로 접는 중...")
                hand.safe()
            except Exception as exc:
                print(f"[tap-hold] safe 오류: {exc}")
        else:
            # 검지를 home(폄)으로 되돌려 마무리 (손가락 다 편 상태)
            try:
                hand.home()
                print("[tap-hold] home(다 폄)으로 복귀")
            except Exception:
                pass
    except Exception as exc:
        print(f"[tap-hold] 오류: {exc}")
        return 1
    finally:
        hand.disconnect()
        print("[tap-hold] 연결 해제")
    return 0


if __name__ == "__main__":
    sys.exit(main())
