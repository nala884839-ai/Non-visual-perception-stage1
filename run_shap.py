"""
run_shap.py — SHAP 기반 피처·모달 중요도 분석 (shap 버전 무관)
=============================================================

무엇을 하나
-----------
- T+A+G+크로스모달(42피처)로 RandomForest 를 학습하고 SHAP 값을 계산.
- shap 반환형이 버전마다 다른 문제를 흡수(리스트 / (n,f,c) 3D / (n,f) 2D 모두 처리).
- 출력 3종:
  1) shap_features_<prefix>.csv : 피처별 평균|SHAP| (중요도 랭킹)
  2) shap_modality_<prefix>.csv : 모달(T/A/G/X)별 중요도 합·비중
  3) shap_perclass_<prefix>.csv : 클래스별 상위 피처 (선택)
- 콘솔에 모달별 비중과 상위 피처를 요약 출력.

왜 중요한가
-----------
ablation 은 "가스를 빼면 정확도가 떨어진다"를 보이고,
SHAP 은 "왜/무엇이 그 판별을 만드는가"를 피처 단위로 보여준다.
가스(G) 모달이 중요도 상위를 차지하면 후각 필요성 주장의 보강 증거가 된다.

주의
----
- 해석 목적이므로 전체 데이터로 적합(성능 주장 아님).
- 트리 모델은 SHAP TreeExplainer 로 정확·빠르게 계산.

사용
----
  python run_shap.py --features feature_table_run1.csv
  python run_shap.py --features f.csv --out-prefix run1 --perclass
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from extract_features import FEATURE_GROUPS
try:
    import stage1_classes as s1c
except ImportError:
    s1c = None

RS = 42
MODAL_LABEL = {"T": "촉각", "A": "음향", "G": "후각(가스)", "X": "크로스모달"}


def feature_columns():
    """T+A+G+X 순서로 피처 컬럼과, 각 컬럼의 모달 태그를 반환."""
    cols, tags = [], []
    for m in ("T", "A", "G", "X"):
        for c in FEATURE_GROUPS[m]:
            cols.append(c)
            tags.append(m)
    return cols, tags


def normalize_shap(sv, n_features):
    """shap 반환형을 피처별 평균|SHAP| (길이 n_features) 로 정규화.

    처리 케이스:
      - list[np.ndarray], 각 (n_samples, n_features)         [구버전 멀티클래스]
      - np.ndarray (n_samples, n_features, n_classes)        [신버전 멀티클래스]
      - np.ndarray (n_samples, n_features)                   [이진/회귀]
    또한 클래스별 (n_features, n_classes) 도 함께 반환.
    """
    if isinstance(sv, list):
        # (n_classes, n_samples, n_features) 로 스택
        arr = np.stack([np.asarray(s) for s in sv], axis=0)  # (C, N, F)
        per_feat = np.abs(arr).mean(axis=(0, 1))             # (F,)
        per_feat_class = np.abs(arr).mean(axis=1).T          # (F, C)
        return per_feat, per_feat_class
    arr = np.asarray(sv)
    if arr.ndim == 3:  # (N, F, C)
        per_feat = np.abs(arr).mean(axis=(0, 2))             # (F,)
        per_feat_class = np.abs(arr).mean(axis=0)            # (F, C)
        return per_feat, per_feat_class
    if arr.ndim == 2:  # (N, F)
        per_feat = np.abs(arr).mean(axis=0)
        return per_feat, per_feat[:, None]
    raise ValueError(f"예상치 못한 SHAP 형태: {arr.shape}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-prefix", default="run1")
    ap.add_argument("--perclass", action="store_true", help="클래스별 상위 피처 CSV 저장")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args(argv)

    import shap  # 여기서 import (미설치면 명확히 에러)
    print(f"shap {shap.__version__}")

    df = pd.read_csv(args.features).dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    cols, tags = feature_columns()
    X = SimpleImputer(strategy="median").fit_transform(df[cols])
    y = df["label"].to_numpy()

    rf = RandomForestClassifier(n_estimators=400, random_state=RS, n_jobs=-1)
    rf.fit(X, y)

    sv = shap.TreeExplainer(rf).shap_values(X)
    per_feat, per_feat_class = normalize_shap(sv, len(cols))
    assert len(per_feat) == len(cols), f"길이 불일치 {len(per_feat)} vs {len(cols)}"

    # 1) 피처별 랭킹
    feat_df = (pd.DataFrame({"feature": cols, "modality": tags, "mean_abs_shap": per_feat})
               .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    feat_df["share_%"] = (feat_df["mean_abs_shap"] / feat_df["mean_abs_shap"].sum() * 100).round(2)
    feat_df.to_csv(f"shap_features_{args.out_prefix}.csv", index=False)

    # 2) 모달별 집계
    mod = (feat_df.groupby("modality")["mean_abs_shap"].sum()
           .reindex(["T", "A", "G", "X"]).fillna(0))
    mod_df = pd.DataFrame({
        "modality": [MODAL_LABEL[m] for m in mod.index],
        "code": mod.index,
        "shap_sum": mod.values,
        "share_%": (mod.values / mod.values.sum() * 100).round(2),
    })
    mod_df.to_csv(f"shap_modality_{args.out_prefix}.csv", index=False)

    # 콘솔 요약
    print("\n" + "=" * 56)
    print("모달별 중요도 (SHAP 합, 비중)")
    for _, r in mod_df.iterrows():
        bar = "█" * int(r["share_%"] / 2)
        print(f"  {r['modality']:<10} {r['share_%']:>5.1f}%  {bar}")
    print("\n" + "=" * 56)
    print(f"상위 {args.topk} 피처")
    for _, r in feat_df.head(args.topk).iterrows():
        print(f"  {MODAL_LABEL[r['modality']]:<10} {r['feature']:<28} {r['share_%']:>5.2f}%")

    # 3) 클래스별 상위 피처 (선택)
    if args.perclass and per_feat_class.shape[1] > 1:
        rows = []
        classes = sorted(np.unique(y))
        for j, lbl in enumerate(classes):
            name = s1c.BY_OBJ_ID[int(lbl) + 1].obj_name if s1c else str(lbl)
            order = np.argsort(per_feat_class[:, j])[::-1][:5]
            for rank, idx in enumerate(order, 1):
                rows.append({"class": name, "rank": rank,
                             "feature": cols[idx], "modality": tags[idx],
                             "mean_abs_shap": float(per_feat_class[idx, j])})
        pd.DataFrame(rows).to_csv(f"shap_perclass_{args.out_prefix}.csv", index=False)
        print(f"\n클래스별 상위 피처 저장: shap_perclass_{args.out_prefix}.csv")

    print(f"\n저장: shap_features_{args.out_prefix}.csv, shap_modality_{args.out_prefix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
