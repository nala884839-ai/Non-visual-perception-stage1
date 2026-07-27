"""
run_collect_stage2a.py
Stage 2A 본 수집: Fruit–Ball 8클래스 (과일 4 + 공 4)
=====================================================================
[현재 설정] day-to-day 없음 → 단일 세션에서 클래스당 30회를 한 번에 수집.
  · --day 를 생략하면 단일 세션 모드(기본 30회/클래스, trial 1..30).
  · --day 1/2/3 을 주면 종전 날짜분산 모드(기본 10회/클래스)로도 동작(하위호환).
  · 1 trial 구조/타이밍/variant 는 config.py + TrialRunner 로 Stage 1 과 동일.
  · variant 기본 'tap'.

수집/분석 흐름 (단일 세션)
--------------------------
  8클래스 × 30 = 240 trials → 5-fold CV + ablation 으로 후각 필요성 입증.
  inter-trial 간격: 과일군(VOC 잔류) 60s, 공군 30s (문서 §2.4).

CLI
---
  # 단일 세션, 8클래스 × 30회 한 번에 (day-to-day 없음)
  python run_collect_stage2a.py

  # 특정 그룹만 / 특정 객체만 / 개체 태그
  python run_collect_stage2a.py --only-group fruit
  python run_collect_stage2a.py --only-name orange --object-tag o2

  # (하위호환) 날짜분산이 필요해지면
  python run_collect_stage2a.py --day 1     # 다른 날 --day 2, --day 3

  # 흐름만 확인
  python run_collect_stage2a.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import config
import stage2a_classes as s2a
from trial_runner import TrialRunner


# 그룹별 inter-trial 간격(초). 과일=VOC 잔류로 길게, 공=짧게 (문서 §2.4)
GROUP_GAP = {"fruit": 60.0, "ball": 30.0}


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage 2A Fruit–Ball 8클래스 본 수집 (날짜 분산)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--day", type=int, default=None, choices=[1, 2, 3],
                    help="(선택) 날짜 분산 수집 시 일차 1/2/3. 생략하면 단일 세션으로 "
                         "클래스당 --per-day 회를 한 번에 수집(trial 1..N, 오프셋 없음)")
    ap.add_argument("--per-day", type=int, default=None,
                    help="클래스당 수집 횟수. --day 지정 시 기본 10, 단일 세션(--day 생략) 시 기본 30")
    ap.add_argument("--variant", choices=list(config.TRIAL_VARIANTS.keys()),
                    default="tap", help="trial variant (기본 tap, Stage 1 과 동일)")
    ap.add_argument("--only-group", choices=s2a.GROUPS, default=None,
                    help="특정 그룹만 수집 (fruit / ball)")
    ap.add_argument("--only-name", choices=[c.obj_name for c in s2a.CLASSES],
                    default=None, help="특정 객체 1개만 수집 (개체 교체 이어받기용)")
    ap.add_argument("--object-tag", default=None,
                    help="개체 식별 태그(예: o1/o2/o3). meta 및 세션명에 남겨 개체 누수 추적")
    ap.add_argument("--session", default=None,
                    help="세션 이름 (미지정 시 자동: stage2a_dayN_날짜)")
    ap.add_argument("--inter-trial-override", type=float, default=None,
                    help="모든 클래스의 trial 간격을 이 값으로 강제 [s]")
    ap.add_argument("--no-pause", action="store_true",
                    help="전환 시 Enter 대기 건너뛰기")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def _pause(msg: str, no_pause: bool) -> None:
    logging.info("=" * 70)
    for line in msg.splitlines():
        logging.info(line)
    logging.info("=" * 70)
    if not no_pause:
        try:
            input(">> 세팅 완료 후 Enter... ")
        except EOFError:
            logging.info("(입력 불가 환경 — 자동 진행)")


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    )

    today = datetime.now().strftime("%Y%m%d")
    tag = f"_{args.object_tag}" if args.object_tag else ""
    single = args.day is None
    # 단일 세션(기본 30) vs 날짜분산(기본 10)
    per = args.per_day if args.per_day is not None else (30 if single else 10)

    if single:
        session = args.session or f"stage2a_single{tag}_{today}"
        tid_offset = 0
    else:
        session = args.session or f"stage2a_day{args.day}{tag}_{today}"
        # trial_id 오프셋: day1→0, day2→per, day3→2*per (Stage 1 과 동일 규칙)
        tid_offset = (args.day - 1) * per

    # 수집 대상 클래스 선택
    classes = list(s2a.CLASSES)
    if args.only_name:
        classes = [c for c in classes if c.obj_name == args.only_name]
    elif args.only_group:
        classes = [c for c in classes if c.group == args.only_group]

    total = len(classes) * per
    mode = "단일 세션" if single else f"DAY {args.day}"
    logging.info("=" * 70)
    logging.info("Stage 2A 본 수집 — %s", mode)
    logging.info("  세션: %s", session)
    logging.info("  클래스: %d개 (%s)", len(classes),
                 ", ".join(c.obj_name for c in classes))
    logging.info("  클래스당 %d회 → 총 %d trials", per, total)
    logging.info("  trial_id 범위: %d ~ %d", tid_offset + 1, tid_offset + per)
    logging.info("  variant=%s", args.variant)
    logging.info("=" * 70)

    if single:
        logging.info("[참고] 단일 세션(day-to-day 없음) → 5-fold CV 로 후각 필요성 입증.")
        logging.info("[참고] 개체 교체 시 --object-tag 로 개체 ID 를 남겨 누수 추적.")
    elif args.day == 1:
        logging.info("[참고] day2/day3 는 반드시 '다른 날짜'에 실행 (day-wise 검증 성립).")

    counter = 0
    prev_group = None
    for c in classes:
        gap = args.inter_trial_override or GROUP_GAP[c.group]

        # 그룹이 바뀌면 그룹 전환 안내
        if c.group != prev_group:
            _pause(f"[그룹 전환] {c.group.upper()} 군 준비\n"
                   f"  - 과일: 동일 숙도/보관/표면온도 유지, 종류별 3개체 이상\n"
                   f"  - 공  : 지그에 고정, tap 접촉점 일정, 종류별 2~3개\n"
                   f"  (문서 §2.3 개체 반복 / §2.4 통제 변수)",
                   args.no_pause)
            prev_group = c.group

        # 객체 세팅 안내
        _pause(f"[객체 세팅] {c.obj_name}  (obj_id={c.obj_id}, label={c.label})\n"
               f"  - {c.display}\n"
               f"  - 배치 위치/접촉점 이전과 동일하게\n"
               f"  - 개체 태그: {args.object_tag or '(미지정)'}",
               args.no_pause)

        for rep in range(1, per + 1):
            counter += 1
            trial_id = tid_offset + rep
            logging.info("-" * 70)
            logging.info("[%d/%d] %s (obj_id=%d)  trial_id=%d  [%s]",
                         counter, total, c.obj_name, c.obj_id, trial_id, mode)
            logging.info("-" * 70)

            runner = TrialRunner(
                session=session,
                obj_id=c.obj_id,
                obj_name=c.obj_name,
                trial_id=trial_id,
                variant=args.variant,
                dry_run=args.dry_run,
            )
            runner.run()

            if rep < per:
                logging.info("Inter-trial 대기 %.0fs (%s)", gap, c.group)
                time.sleep(gap)

    logging.info("=" * 70)
    logging.info("%s 수집 완료: %d trials → session=%s", mode, total, session)
    if single:
        logging.info("8클래스 × %d회 = %d trials 확보", per, total)
        logging.info("다음 단계: python extract_features_v2b_profile.py "
                     "--data-root data/raw/%s --stage 2a --out feature_table_stage2a.csv", session)
    elif args.day < 3:
        logging.info("다음: 다른 날짜에  python run_collect_stage2a.py --day %d  실행", args.day + 1)
    else:
        logging.info("3 day 수집 완료 → 8클래스 × 30회 = 240 trials 목표")
    logging.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
