"""
run_experiment.py
④ Amazing Hand Action ablation + Teensy 센서 통합 실험 메인 엔트리
=====================================================================

CLI 실행 예시
-------------
  # 단일 trial (tap only)
  python run_experiment.py --obj-id 1 --obj-name aluminum --trial-id 1 --variant tap

  # Ablation 전체 (한 물체에 3가지 variant 각 N회 반복)
  python run_experiment.py --obj-id 1 --obj-name aluminum --ablation --repeats 3

  # 단일 variant 반복 (한 물체에 tap 만 N회 — tap-only 데이터 수집)
  python run_experiment.py --obj-id 1 --obj-name aluminum --variant tap --repeat-single --repeats 30

  # 하드웨어 없이 타이밍만 확인
  python run_experiment.py --obj-id 1 --obj-name test --trial-id 1 --variant tap --dry-run

하드웨어 체크리스트
--------------------
  1. Amazing Hand 서보 8개 전원 + COM3 연결
  2. Teensy 4.1 펌웨어 업로드 (firmware/teensy_sensor_logger/)
     - WowSkin × 5 (TCA9548A mux 0~4)
     - Knowles BU × 2 (A0/A1)
     - ENS160 (I2C 0x53) + BME280 (0x77) + BME688 (0x76)
  3. Teensy COM4 연결 (config.py 수정 가능)
  4. conda activate nonvisual
  5. pip install rustypot pyserial numpy
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import config
from trial_runner import TrialRunner


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Non-visual multimodal object recognition — experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 필수
    ap.add_argument("--obj-id",   type=int, required=True, help="물체 ID (예: 1)")
    ap.add_argument("--obj-name", required=True, help="물체 이름 (예: aluminum)")

    # 모드 선택
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial-id", type=int, help="단일 trial 모드: trial 번호")
    group.add_argument("--ablation", action="store_true",
                       help="Ablation 모드: 3 variant × --repeats 회 실행")
    group.add_argument("--repeat-single", action="store_true",
                       help="단일 variant 반복 모드: --variant 하나를 --repeats 회 실행")

    # 공통 옵션
    ap.add_argument("--variant", choices=list(config.TRIAL_VARIANTS.keys()),
                    default="tap_press_rub",
                    help="단일 trial / 단일 variant 반복 모드에서 사용할 variant")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Ablation / 단일 variant 반복 모드에서 반복 횟수")
    ap.add_argument("--session", default=None,
                    help="세션 이름 (미지정 시 현재 시각 기반 자동 생성)")
    ap.add_argument("--inter-trial-sleep", type=float, default=30.0,
                    help="trial 간 대기 시간 [s]. 기본 30s (문서 §8.1 baseline 복귀). "
                         "휘발성 용제(ethanol/IPA 등)는 잔류 VOC 때문에 60s 이상 권장 (문서 §15)")

    ap.add_argument("--dry-run", action="store_true",
                    help="하드웨어 없이 타이밍만 시뮬레이션")
    ap.add_argument("-v", "--verbose", action="store_true")

    return ap


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    )

    session = args.session or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ------------------------------------------------------------------
    # 단일 trial
    # ------------------------------------------------------------------
    if args.trial_id is not None:
        runner = TrialRunner(
            session=session,
            obj_id=args.obj_id,
            obj_name=args.obj_name,
            trial_id=args.trial_id,
            variant=args.variant,
            dry_run=args.dry_run,
        )
        runner.run()
        return 0

    # ------------------------------------------------------------------
    # 단일 variant 반복: --variant 하나를 N회
    # ------------------------------------------------------------------
    if args.repeat_single:
        total = args.repeats
        logging.info("단일 variant 반복: obj%02d_%s, variant=%s × %d회",
                     args.obj_id, args.obj_name, args.variant, total)

        for rep in range(1, total + 1):
            logging.info("-" * 70)
            logging.info("[%d/%d] variant=%s  trial_id=%d",
                         rep, total, args.variant, rep)
            logging.info("-" * 70)

            runner = TrialRunner(
                session=session,
                obj_id=args.obj_id,
                obj_name=args.obj_name,
                trial_id=rep,
                variant=args.variant,
                dry_run=args.dry_run,
            )
            runner.run()

            if rep < total:
                logging.info("Inter-trial 대기 %.1fs", args.inter_trial_sleep)
                time.sleep(args.inter_trial_sleep)

        logging.info("단일 variant 반복 완료: %d trials → session=%s", total, session)
        return 0
    variants = list(config.TRIAL_VARIANTS.keys())
    total = len(variants) * args.repeats
    trial_counter = 0

    logging.info("Ablation 시작: obj%02d_%s, %d variants × %d repeats = %d trials",
                 args.obj_id, args.obj_name, len(variants), args.repeats, total)

    for variant in variants:
        for rep in range(1, args.repeats + 1):
            trial_counter += 1
            trial_id = trial_counter

            logging.info("-" * 70)
            logging.info("[%d/%d] variant=%s  repeat=%d/%d  trial_id=%d",
                         trial_counter, total, variant, rep, args.repeats, trial_id)
            logging.info("-" * 70)

            runner = TrialRunner(
                session=session,
                obj_id=args.obj_id,
                obj_name=args.obj_name,
                trial_id=trial_id,
                variant=variant,
                dry_run=args.dry_run,
            )
            runner.run()

            # 마지막 trial 아니면 inter-trial 대기
            if trial_counter < total:
                logging.info("Inter-trial 대기 %.1fs", args.inter_trial_sleep)
                time.sleep(args.inter_trial_sleep)

    logging.info("Ablation 완료: %d trials → session=%s", total, session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
