"""
run_collect_stage2b.py
Stage 2B 본 수집: Bio-like Surface 3클래스 (피부 / Dragon Skin 10 / 생닭)
=====================================================================
Stage 1 의 run_collect_9class.py 구조를 그대로 따른다:
  · day 단위 실행(--day 1/2/3), 각 day 클래스당 --per-day(기본 10) 회
  · trial_id 오프셋 → day1:1..10, day2:11..20, day3:21..30
  · 3 day = 클래스당 30 trial = 3 × 30 = 90 trials (문서 §3.3)
  · 1 trial 구조/타이밍/variant 는 config.py + TrialRunner 로 Stage 1 과 동일
  · variant 기본 'tap'

Stage 2B 고유의 안전·윤리 통제(문서 §3.4, §7)
--------------------------------------------
  · 생닭가슴살(chicken)은 raw poultry → 전용 장갑/트레이/일회용 필름, 접촉 후 소독,
    측정 후 긴 purge(기본 90s). 스크립트가 클래스 전환 시 안내를 강제 출력한다.
  · 사람 피부(skin)는 자발적 동의·접촉 부위 고정·접촉 힘 제한·소독 안내를 출력.
    ※ 사람 대상/생체시료 실험은 기관·지도교수 안전/윤리(IRB 해당 여부) 확인 대상 —
       스크립트는 이를 확인했다는 전제하에만 진행하도록 시작 시 경고한다.
  · 기본 조건은 "clean/dry 또는 생리식염수·인공땀" (문서 §3.4). '침'은 강한 후각
    shortcut 이므로 main class 가 아니라 별도 control 로만 다룬다(스크립트 미포함).

CLI
---
  python run_collect_stage2b.py

  # 특정 클래스만
  python run_collect_stage2b.py --only-name chicken

  # (하위호환) 날짜분산
  python run_collect_stage2b.py --day 1

  # 흐름만 확인
  python run_collect_stage2b.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import config
import stage2b_classes as s2b
from trial_runner import TrialRunner


# 클래스별 inter-trial 간격(초). 생닭은 잔류 냄새로 긴 purge (문서 §3.4)
NAME_GAP = {"skin": 45.0, "dragonskin": 45.0, "chicken": 90.0}

# 클래스별 세팅/안전 안내 (문서 §3.4)
SETUP_NOTE = {
    "skin": ("- 참여자 자발적 동의 확인 / 접촉 부위 고정 / 접촉 힘 제한\n"
             "- 기본 조건: clean·dry 또는 생리식염수·인공땀 (침 사용 금지, 별도 control)\n"
             "- 실험 전후 접촉 부위 소독"),
    "dragonskin": ("- 동일 두께·면적 패드 3개 이상, 표면 이물질 제거\n"
                   "- 경화 완료(cure) 상태 확인 (미경화 잔류는 후각 shortcut)"),
    "chicken": ("- raw poultry: 전용 장갑·전용 트레이·일회용 필름 사용\n"
                "- 3개 이상 조각 또는 다른 포장 lot, 수분/온도/보관시간 기록\n"
                "- 측정 후 접촉면 소독, 충분한 purge/recovery (기본 90s)"),
}


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage 2B Bio-like Surface 3클래스 본 수집 (날짜 분산)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--day", type=int, default=None, choices=[1, 2, 3],
                    help="(선택) 날짜 분산 수집 시 일차 1/2/3. 생략하면 단일 세션으로 "
                         "클래스당 --per-day 회를 한 번에 수집(trial 1..N)")
    ap.add_argument("--per-day", type=int, default=None,
                    help="클래스당 수집 횟수. --day 지정 시 기본 10, 단일 세션 시 기본 30")
    ap.add_argument("--variant", choices=list(config.TRIAL_VARIANTS.keys()),
                    default="tap", help="trial variant (기본 tap, Stage 1 과 동일)")
    ap.add_argument("--only-name", choices=[c.obj_name for c in s2b.CLASSES],
                    default=None, help="특정 클래스 1개만 수집")
    ap.add_argument("--object-tag", default=None,
                    help="개체/lot 식별 태그. meta 및 세션명에 남겨 누수 추적")
    ap.add_argument("--session", default=None,
                    help="세션 이름 (미지정 시 자동: stage2b_dayN_날짜)")
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
    per = args.per_day if args.per_day is not None else (30 if single else 10)
    if single:
        session = args.session or f"stage2b_single{tag}_{today}"
        tid_offset = 0
    else:
        session = args.session or f"stage2b_day{args.day}{tag}_{today}"
        tid_offset = (args.day - 1) * per

    classes = list(s2b.CLASSES)
    if args.only_name:
        classes = [c for c in classes if c.obj_name == args.only_name]

    total = len(classes) * per
    mode = "단일 세션" if single else f"DAY {args.day}"
    logging.info("=" * 70)
    logging.info("Stage 2B 본 수집 — %s", mode)
    logging.info("  세션: %s", session)
    logging.info("  클래스: %d개 (%s)", len(classes),
                 ", ".join(c.obj_name for c in classes))
    logging.info("  클래스당 %d회 → 총 %d trials", per, total)
    logging.info("  trial_id 범위: %d ~ %d", tid_offset + 1, tid_offset + per)
    logging.info("=" * 70)

    # 시작 시 안전/윤리 확인 경고 (문서 §7)
    _pause("[안전·윤리 확인 — 필수]\n"
           "  - 사람 피부/생체시료 사용에 대한 기관·지도교수 안전/윤리(IRB 해당 여부) 확인 완료?\n"
           "  - 생닭 교차오염 방지 물품(장갑/트레이/필름/소독) 준비 완료?\n"
           "  이 두 가지가 확인된 경우에만 진행하세요.",
           args.no_pause)

    if single:
        logging.info("[참고] 단일 세션(day-to-day 없음) → 5-fold CV 로 분석.")
    elif args.day == 1:
        logging.info("[참고] day2/day3 는 반드시 '다른 날짜'에 실행.")

    counter = 0
    for c in classes:
        gap = args.inter_trial_override or NAME_GAP[c.obj_name]
        bio = "  ⚠ BIOHAZARD (raw poultry)\n" if c.biohazard else ""
        _pause(f"[클래스 세팅] {c.obj_name}  (obj_id={c.obj_id}, label={c.label})\n"
               f"{bio}  {c.display}\n"
               f"{SETUP_NOTE[c.obj_name]}\n"
               f"  개체/lot 태그: {args.object_tag or '(미지정)'}",
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
                logging.info("Inter-trial(purge) 대기 %.0fs (%s)", gap, c.obj_name)
                time.sleep(gap)

    logging.info("=" * 70)
    logging.info("%s 수집 완료: %d trials → session=%s", mode, total, session)
    if single:
        logging.info("3클래스 × %d회 = %d trials 확보", per, total)
        logging.info("다음 단계: python extract_features_v2b_profile.py "
                     "--data-root data/raw/%s --stage 2b --out feature_table_stage2b.csv", session)
    elif args.day < 3:
        logging.info("다음: 다른 날짜에  python run_collect_stage2b.py --day %d  실행", args.day + 1)
    else:
        logging.info("3 day 수집 완료 → 3클래스 × 30회 = 90 trials 목표")
        logging.info("다음 단계: python extract_features_stage2.py --stage 2b --session <각 세션>")
    logging.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
