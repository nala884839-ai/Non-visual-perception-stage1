"""
collect_stage1.py
① Stage 1 9-class 단일 세션 수집 오케스트레이터
================================================

ML_설계_정리.md 개정판 기준:
- 9클래스 × 30 trial = 270 trial, **단일 세션** (날짜 분산 없음 → Stage 2)
- VOC 클래스(ethanol/acetone)는 trial 간격 60s+ (알코올 잔류 baseline 복귀)
- empty 클래스는 일반 간격(기본 30s)

기존 trial_runner.TrialRunner 를 그대로 호출하므로 하드웨어/마커/저장 로직은 재사용.
클래스 순서와 obj_id/obj_name 매핑만 stage1_classes 에서 가져온다.

주의
----
- 이 스크립트는 실제 하드웨어(Hand/Teensy/Stage)가 연결된 환경에서 실행한다.
- 한 클래스 끝나면 컵 교체/세척·킴테크 교체 프롬프트로 일시정지(안전).
- 중단 후 재개: --start-class 로 특정 클래스부터 이어서 수집.

CLI
---
  python collect_stage1.py --variant tap_press_rub --trials 30
  python collect_stage1.py --trials 30 --start-class 4   # ceramic_empty 부터 재개
  python collect_stage1.py --dry-run --trials 2          # 타이밍만 점검
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import config
from trial_runner import TrialRunner

try:
    import stage1_classes as s1c
except ImportError:
    print("stage1_classes.py 를 같은 폴더에 두세요.", file=sys.stderr)
    raise

LOG = logging.getLogger("collect_stage1")

# VOC vs 일반 trial 간격 (초)
VOC_INTER_TRIAL_S = 60.0
PLAIN_INTER_TRIAL_S = 30.0


def collect(session: str, variant: str, trials_per_class: int,
            start_class: int, dry_run: bool) -> None:
    classes = [c for c in s1c.CLASSES if c.obj_id >= start_class]
    total = len(classes) * trials_per_class
    LOG.info("Stage 1 수집 시작 — 세션=%s, %d 클래스 × %d trial = %d trial (단일 세션)",
             session, len(classes), trials_per_class, total)
    LOG.info("variant=%s, VOC 간격=%.0fs / 일반 간격=%.0fs",
             variant, VOC_INTER_TRIAL_S, PLAIN_INTER_TRIAL_S)

    done = 0
    for c in classes:
        gap = VOC_INTER_TRIAL_S if c.is_voc else PLAIN_INTER_TRIAL_S
        LOG.info("=" * 70)
        LOG.info("클래스 obj%02d_%s (label=%d, %s/%s, VOC=%s)",
                 c.obj_id, c.obj_name, c.label, c.material, c.content, c.is_voc)
        LOG.info("→ 컵/내용물 세팅 확인 후 진행 (킴테크 교체, 컵 건조 등)")
        LOG.info("=" * 70)
        if not dry_run:
            input("    준비되면 Enter 를 눌러 이 클래스 수집을 시작하세요... ")

        for trial_id in range(1, trials_per_class + 1):
            done += 1
            LOG.info("-" * 70)
            LOG.info("[%d/%d] obj%02d_%s trial_%03d",
                     done, total, c.obj_id, c.obj_name, trial_id)
            runner = TrialRunner(
                session=session,
                obj_id=c.obj_id,
                obj_name=c.obj_name,
                trial_id=trial_id,
                variant=variant,
                dry_run=dry_run,
            )
            runner.run()
            if trial_id < trials_per_class:
                LOG.info("Inter-trial 대기 %.0fs (%s)", gap,
                         "VOC 잔류 복귀" if c.is_voc else "일반")
                time.sleep(gap if not dry_run else min(gap, 1.0))

    LOG.info("Stage 1 수집 완료: %d trial → 세션=%s", done, session)
    LOG.info("다음 단계: python extract_features.py --session %s", session)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1 9-class 단일 세션 수집")
    ap.add_argument("--variant", choices=list(config.TRIAL_VARIANTS.keys()),
                    default="tap",
                    help="trial phase 구성 (기본: tap — Stage 1 후각 입증 목표, "
                         "press/rub 없이 tap 접촉만으로 T/A 블록 확보)")
    ap.add_argument("--trials", type=int, default=30, help="클래스당 trial 수 (기본 30)")
    ap.add_argument("--start-class", type=int, default=1,
                    help="이 obj_id 부터 재개 (1~9)")
    ap.add_argument("--session", default=None,
                    help="세션 이름 (미지정 시 자동 — 단일 세션 보장 위해 한 번만 생성)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    session = args.session or f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    collect(session, args.variant, args.trials, args.start_class, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
