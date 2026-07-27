"""
analyze_stage2a_subgroup.py
Stage 2A 서브그룹(과일군/공군) 지배 모달 분석
=====================================================================
run_ml.py 의 실험 엔진(run_experiment, MODAL_COMBOS, CLASSIFIERS, FUSIONS)을
그대로 재사용하여, 문서 §5.3 Fig.4 및 가설 A-H1/A-H2/A-H3 을 위한
서브그룹별 7-조합 성능표를 만든다.

산출(문서 §2.2 / §5.3 대응)
--------------------------
  · 전체 8-class    : T+A+O 가 가장 안정적인가? (A-H3)
  · 과일군 4-class  : O 추가 시 오렌지↔자몽/사과↔배 혼동 감소? (A-H1)
  · 공군 4-class    : O 단독은 약하고 T+A 가 재질·구조를 반영? (A-H2)

각 (subset × combo) 에서 fusion×classifier 최고 정확도를 대표값으로 뽑아
Stage 1 combo7_summary.csv 와 같은 형식의 표를 만든다.
※ 여기서 O(후각)는 run_ml 의 모달 코드 'G' 에 해당(ENS160+BME688).

CLI
---
  python analyze_stage2a_subgroup.py --features feature_table_stage2a.csv \
      --out subgroup_stage2a.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import run_ml  # 실험 엔진 그대로 재사용

LOG = logging.getLogger("analyze_stage2a_subgroup")


def _best_acc_per_combo(df: pd.DataFrame) -> dict:
    """subset df 에 대해 7-조합 각각의 (fusion×classifier) 최고 5-fold 정확도."""
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    y = df["label"].to_numpy()
    n_per = pd.Series(y).value_counts()
    n_splits = min(run_ml.N_SPLITS, int(n_per.min()))
    if n_splits < 2:
        raise ValueError(f"클래스당 표본 부족 (min={n_per.min()})")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=run_ml.RANDOM_STATE)

    out = {}
    for combo in run_ml.MODAL_COMBOS:
        best = -1.0
        for fusion in run_ml.FUSIONS:
            for clf in run_ml.CLASSIFIERS:
                r = run_ml.run_experiment(df, combo, fusion, clf, skf)
                if r["acc_mean"] == r["acc_mean"] and r["acc_mean"] > best:
                    best = r["acc_mean"]
        out[combo] = round(100.0 * best, 1) if best >= 0 else float("nan")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2A 서브그룹 지배 모달 분석")
    ap.add_argument("--features", required=True, help="feature_table_stage2a.csv")
    ap.add_argument("--out", default="subgroup_stage2a.csv")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_csv(args.features)
    if "group" not in df.columns:
        raise ValueError("feature_table 에 'group' 컬럼이 없습니다. "
                         "extract_features_stage2.py --stage 2a 로 추출했는지 확인.")

    subsets = {
        "all_8class": df,
        "fruit_4class": df[df["group"] == "fruit"],
        "ball_4class": df[df["group"] == "ball"],
    }

    rows = []
    for name, sub in subsets.items():
        n_class = sub["label"].nunique()
        LOG.info("[%s] trial=%d, class=%d", name, len(sub), n_class)
        try:
            best = _best_acc_per_combo(sub)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s 분석 실패: %s", name, exc)
            continue
        row = {"subset": name, "n_trial": len(sub), "n_class": int(n_class)}
        row.update(best)
        rows.append(row)
        LOG.info("  %s", {k: best[k] for k in run_ml.MODAL_COMBOS})

    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    LOG.info("저장: %s", args.out)
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
