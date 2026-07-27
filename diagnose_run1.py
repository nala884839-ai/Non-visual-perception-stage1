"""
diagnose_run1.py — 후각 이득이 '에탄올↔아세톤 분리'에서 나오는지 + 시간누출 점검
==============================================================================

세 가지를 확인한다.
1) T+A vs T+A+G (early RF) 혼동행렬 → 가스 추가로 어떤 혼동이 사라지는지
2) 내용물(empty/ethanol/acetone) 단위 정확도 → 가스가 ethanol↔acetone 을 살리는지
3) 시간순서 누출 점검: 무작위 5-fold vs '각 클래스 앞 trial 학습 / 뒤 trial 검증'(시간분할)
   두 정확도 차이가 크면 → 단일세션 순차수집으로 인한 ordering leakage 신호.

사용:
  python diagnose_run1.py --features feature_table_run1.csv
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
import stage1_classes as s1c

RS = 42
CONTENTS = ["empty", "ethanol", "acetone"]
MATERIALS = ["glass", "ceramic", "plastic"]


def _pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1))])


def _cols(combo):
    blocks = {"T": FEATURE_GROUPS["T"], "A": FEATURE_GROUPS["A"], "G": FEATURE_GROUPS["G"]}
    cols = []
    for m in combo.split("+"):
        cols += blocks[m]
    if len(combo.split("+")) >= 2:
        cols += FEATURE_GROUPS["X"]
    return cols


def label_name(lbl):
    k = s1c.BY_OBJ_ID.get(int(lbl) + 1)
    return k.obj_name if k else str(lbl)


def confusion(df, combo, skf):
    X = df[_cols(combo)].to_numpy(float)
    y = df["label"].astype(int).to_numpy()
    pred = cross_val_predict(_pipe(), X, y, cv=skf)
    labels = sorted(np.unique(y))
    cm = pd.crosstab(pd.Series([label_name(i) for i in y], name="true"),
                     pd.Series([label_name(i) for i in pred], name="pred"))
    acc = float(np.mean(pred == y))
    return cm, acc, y, pred


def content_accuracy(df, y, pred):
    """내용물 축만 맞췄는지 (재질 무시)."""
    true_c = df["content"].to_numpy()
    pred_c = np.array([s1c.BY_OBJ_ID[int(p) + 1].content for p in pred])
    out = {}
    for c in CONTENTS:
        m = true_c == c
        if m.sum():
            out[c] = float(np.mean(pred_c[m] == c))
    return out


def temporal_leakage_check(df):
    """무작위 5-fold vs 시간분할(각 클래스 앞80% 학습/뒤20% 검증) 정확도 비교."""
    cols = _cols("T+A+G")
    X = df[cols].to_numpy(float)
    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    rand_acc = float(np.mean(cross_val_predict(_pipe(), X, y, cv=skf) == y))

    # 시간분할: trial_id 기준 각 클래스 뒤 20% 를 검증으로
    df2 = df.reset_index(drop=True)
    test_idx = []
    for lbl, g in df2.groupby("label"):
        g = g.sort_values("trial_id")
        k = max(1, int(round(len(g) * 0.2)))
        test_idx += list(g.index[-k:])
    test_mask = df2.index.isin(test_idx)
    tr, te = ~test_mask, test_mask
    pipe = _pipe(); pipe.fit(X[tr], y[tr])
    temporal_acc = float(np.mean(pipe.predict(X[te]) == y[te]))
    return rand_acc, temporal_acc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-prefix", default="run1")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.features).dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    pfx = args.out_prefix

    print("=" * 64)
    print("① 혼동행렬 (early RF, 5-fold out-of-fold 예측)")
    content_rows = []
    cm_store = {}
    for combo in ("T+A", "T+A+G"):
        cm, acc, y, pred = confusion(df, combo, skf)
        cm_store[combo] = (cm, acc, y, pred)
        print(f"\n--- {combo}  (acc={acc:.3f}) ---")
        print(cm.to_string())
        cacc = content_accuracy(df, y, pred)
        print("  내용물별 정확도:", {k: round(v, 3) for k, v in cacc.items()})
        # 혼동행렬 CSV 저장 (combo 별)
        cm.to_csv(f"diagnose_confusion_{combo.replace('+','')}_{pfx}.csv")

    print("\n" + "=" * 64)
    print("② 핵심 질문: 가스 추가로 ethanol/acetone 내용물 정확도가 오르는가?")
    _, _, y0, p0 = cm_store["T+A"]
    _, _, y1, p1 = cm_store["T+A+G"]
    c0 = content_accuracy(df, y0, p0)
    c1 = content_accuracy(df, y1, p1)
    for c in CONTENTS:
        if c in c0 and c in c1:
            print(f"  {c:<8}: T+A {c0[c]:.3f}  →  T+A+G {c1[c]:.3f}  (Δ {c1[c]-c0[c]:+.3f})")
            content_rows.append({"content": c, "acc_TA": round(c0[c], 4),
                                 "acc_TAG": round(c1[c], 4),
                                 "delta": round(c1[c] - c0[c], 4)})

    print("\n" + "=" * 64)
    print("③ 시간순서 누출 점검 (단일세션 순차수집 검증)")
    rand_acc, temporal_acc = temporal_leakage_check(df)
    gap = rand_acc - temporal_acc
    print(f"  무작위 5-fold 정확도 : {rand_acc:.3f}")
    print(f"  시간분할 정확도      : {temporal_acc:.3f}  (각 클래스 뒤 20% trial 검증)")
    print(f"  차이(gap)            : {gap:+.3f}")
    # 문턱 0.10 로 하향(0.13~0.15 는 무시 못 함)
    leak = gap > 0.10
    if leak:
        print("  ⚠ gap 있음 → 시간드리프트 누출 존재. 무작위 5-fold 는 낙관적, "
              "시간분할을 일반화 하한으로 병기 권장 (Stage 2 날짜분산+LODO 필요).")
    else:
        print("  ✓ gap 작음 → 순차수집으로 인한 누출은 크지 않음.")

    # ②·③ 요약 CSV 저장
    pd.DataFrame(content_rows).to_csv(f"diagnose_content_delta_{pfx}.csv", index=False)
    pd.DataFrame([{
        "random_5fold_acc": round(rand_acc, 4),
        "temporal_split_acc": round(temporal_acc, 4),
        "leakage_gap": round(gap, 4),
        "leakage_flag": bool(leak),
        "TA_acc": round(cm_store["T+A"][1], 4),
        "TAG_acc": round(cm_store["T+A+G"][1], 4),
    }]).to_csv(f"diagnose_summary_{pfx}.csv", index=False)

    print(f"\n저장: diagnose_confusion_TA_{pfx}.csv, diagnose_confusion_TAG_{pfx}.csv, "
          f"diagnose_content_delta_{pfx}.csv, diagnose_summary_{pfx}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
