"""
temporal_ablation.py — 시간분할(정직한 검증)에서도 후각(G) 기여가 유의한지 확인
==============================================================================

무작위 5-fold 는 단일세션 순차수집의 시간드리프트를 누출로 이용해 정확도를 부풀린다.
여기서는 '각 클래스의 뒤 20% trial' 을 검증셋으로 고정(시간분할)한 뒤,
  Full(T+A+G) vs T+A  의 정확도 차이를 본다.
시간분할에서도 Full > T+A 면 → 후각 기여는 드리프트 누출이 아니라 진짜다.

또한 seed 를 바꿔가며 무작위 5-fold 를 반복해, 시간분할 정확도가
무작위 분포의 어디쯤인지(누출 크기)도 참고로 보여준다.

사용:
  python temporal_ablation.py --features feature_table_run1.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from extract_features import FEATURE_GROUPS

RS = 42


def _pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))])


def _cols(combo):
    cols = []
    for m in combo.split("+"):
        cols += FEATURE_GROUPS[m]
    if len(combo.split("+")) >= 2:
        cols += FEATURE_GROUPS["X"]
    return cols


def temporal_split(df):
    """각 클래스 trial_id 기준 뒤 20% 를 검증셋으로."""
    df = df.reset_index(drop=True)
    test_idx = []
    for _, g in df.groupby("label"):
        g = g.sort_values("trial_id")
        k = max(1, int(round(len(g) * 0.2)))
        test_idx += list(g.index[-k:])
    mask = df.index.isin(test_idx)
    return ~mask, mask


def acc_temporal(df, combo, tr, te):
    X = df[_cols(combo)].to_numpy(float)
    y = df["label"].astype(int).to_numpy()
    p = _pipe().fit(X[tr], y[tr]).predict(X[te])
    return float(np.mean(p == y[te]))


def acc_random(df, combo, seed):
    X = df[_cols(combo)].to_numpy(float)
    y = df["label"].astype(int).to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_predict(_pipe(), X, y, cv=skf) == y))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-prefix", default="run1")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.features).dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    tr, te = temporal_split(df)

    print("=" * 60)
    print("시간분할(정직한 검증)에서의 후각 ablation")
    print(f"  검증셋: 각 클래스 뒤 20% trial ({int(te.sum())}개)")
    full_t = acc_temporal(df, "T+A+G", tr, te)
    ta_t = acc_temporal(df, "T+A", tr, te)
    delta = full_t - ta_t
    print(f"  Full (T+A+G) : {full_t:.3f}")
    print(f"  T+A (후각 제거): {ta_t:.3f}")
    print(f"  후각 기여 Δ   : {delta:+.3f}")
    survives = delta > 0.05
    if survives:
        print("  ✓ 시간분할에서도 후각이 크게 기여 → 후각 필요성은 누출이 아닌 진짜.")
    else:
        print("  ⚠ 시간분할에서 후각 기여가 작음 → 재검토 필요.")

    print("\n" + "=" * 60)
    print(f"참고: 무작위 5-fold {args.seeds}회 반복 vs 시간분할 (Full T+A+G)")
    rnd = [acc_random(df, "T+A+G", s) for s in range(args.seeds)]
    gap = float(np.mean(rnd)) - full_t
    print(f"  무작위 5-fold : {np.mean(rnd):.3f} ± {np.std(rnd):.3f}  "
          f"(범위 {min(rnd):.3f}~{max(rnd):.3f})")
    print(f"  시간분할      : {full_t:.3f}")
    print(f"  누출 추정 gap : {gap:+.3f}")
    print("  → 이 gap 만큼 무작위 5-fold 가 낙관적. 논문엔 두 값을 함께 보고 권장.")

    # CSV 저장
    pd.DataFrame([{
        "temporal_full_TAG": round(full_t, 4),
        "temporal_TA": round(ta_t, 4),
        "temporal_olfactory_delta": round(delta, 4),
        "olfactory_survives_temporal": bool(survives),
        "random_5fold_mean": round(float(np.mean(rnd)), 4),
        "random_5fold_std": round(float(np.std(rnd)), 4),
        "random_5fold_min": round(float(min(rnd)), 4),
        "random_5fold_max": round(float(max(rnd)), 4),
        "leakage_gap": round(gap, 4),
        "n_test_trials": int(te.sum()),
        "n_seeds": args.seeds,
    }]).to_csv(f"temporal_ablation_{args.out_prefix}.csv", index=False)
    print(f"\n저장: temporal_ablation_{args.out_prefix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
