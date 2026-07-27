"""
run_ml.py
③+④ feature_table.csv → 63 실험 + ablation 분석
================================================

설계 (ML_설계_정리.md 개정판)
------------------------------
- 모달 조합 7 : T, A, G, T+A, T+G, A+G, T+A+G
- 융합 전략 3 : early / feature / late
- 분류기 3   : RandomForest / SVM(RBF) / MLP
- 검증       : **Stratified 5-fold CV 만** (LODO 는 Stage 2)
- ablation   : Full(T+A+G) vs Full−G / Full−T / Full−A → fold별 paired t-test
                → 후각(G) 제거 시 정확도 유의 하락 = 후각 필요성 입증(핵심)

융합 전략 정의
--------------
- early   : 모든 모달 피처를 그대로 concat → 1개 분류기 (표준화만)
- feature : 모달별 표준화 후 concat (모달별 전처리 분리) → 1개 분류기
- late    : 모달별로 분류기를 따로 학습 → 예측확률 평균(soft voting)
  ※ 단일 모달 조합(T/A/G)에서는 세 전략이 입력상 동일 → 결과 중복(정상).

크로스모달 피처(X)는 ≥2 모달 조합에만 포함(단일 모달엔 미포함).

출력
----
  results_<session>.csv      : 63행 (combo×fusion×clf, accuracy/F1 mean±std)
  ablation_<session>.csv     : Full vs Full−X paired t-test
  (shap 설치 시) shap_<...>.csv 상위 피처 중요도

CLI
---
  python run_ml.py --features feature_table_xxx.csv
  python run_ml.py --features f.csv --out-prefix stage1
"""

from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

from extract_features import FEATURE_GROUPS

warnings.filterwarnings("ignore")
LOG = logging.getLogger("run_ml")

N_SPLITS = 5
RANDOM_STATE = 42

MODAL_COMBOS = ["T", "A", "G", "T+A", "T+G", "A+G", "T+A+G"]
FUSIONS = ["early", "feature", "late"]


# =============================================================================
# 분류기 팩토리
# =============================================================================

def make_classifier(name: str):
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                             random_state=RANDOM_STATE, early_stopping=True)
    raise ValueError(name)


CLASSIFIERS = ["RF", "SVM", "MLP"]


# =============================================================================
# 피처 컬럼 선택
# =============================================================================

def modalities_in_combo(combo: str) -> List[str]:
    return combo.split("+")


def combo_feature_blocks(combo: str) -> Dict[str, List[str]]:
    """조합에 들어가는 모달별 피처 컬럼 딕셔너리. ≥2 모달이면 X(크로스모달) 포함."""
    mods = modalities_in_combo(combo)
    blocks = {m: list(FEATURE_GROUPS[m]) for m in mods}
    if len(mods) >= 2:
        blocks["X"] = list(FEATURE_GROUPS["X"])
    return blocks


def _pipe(clf_name: str) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", make_classifier(clf_name)),
    ])


# =============================================================================
# 융합 전략별 fold 학습/평가
# =============================================================================

def _eval_concat(X: np.ndarray, y: np.ndarray, clf_name: str,
                 skf: StratifiedKFold) -> Tuple[np.ndarray, np.ndarray]:
    """early: 모든 모달 피처를 그대로 concat → impute+전역표준화 → 1개 분류기."""
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        pipe = _pipe(clf_name)
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.array(accs), np.array(f1s)


def _eval_feature(blocks: Dict[str, np.ndarray], y: np.ndarray, clf_name: str,
                  skf: StratifiedKFold) -> Tuple[np.ndarray, np.ndarray]:
    """feature-level: 모달별 (impute→표준화→PCA 95%) 후 concat → 1개 분류기.
    early 와 달리 모달별 전처리/차원축소를 분리 → 수치적으로 구분됨."""
    accs, f1s = [], []
    first = next(iter(blocks.values()))
    for tr, te in skf.split(first, y):
        Xtr_parts, Xte_parts = [], []
        for Xb in blocks.values():
            imp = SimpleImputer(strategy="median")
            sc = StandardScaler()
            btr = sc.fit_transform(imp.fit_transform(Xb[tr]))
            bte = sc.transform(imp.transform(Xb[te]))
            ncomp = max(1, min(btr.shape[1], btr.shape[0] - 1))
            pca = PCA(n_components=ncomp, random_state=RANDOM_STATE)
            Xtr_parts.append(pca.fit_transform(btr))
            Xte_parts.append(pca.transform(bte))
        Xtr = np.hstack(Xtr_parts); Xte = np.hstack(Xte_parts)
        clf = make_classifier(clf_name)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.array(accs), np.array(f1s)


def _eval_late(blocks: Dict[str, np.ndarray], y: np.ndarray, clf_name: str,
               skf: StratifiedKFold) -> Tuple[np.ndarray, np.ndarray]:
    """late fusion: 모달별 분류기 학습 → 예측확률 평균(soft voting)."""
    accs, f1s = [], []
    classes = np.unique(y)
    for tr, te in skf.split(next(iter(blocks.values())), y):
        proba_sum = np.zeros((len(te), len(classes)))
        n_used = 0
        for _, Xb in blocks.items():
            pipe = _pipe(clf_name)
            try:
                pipe.fit(Xb[tr], y[tr])
                proba = pipe.predict_proba(Xb[te])
                # 클래스 정렬 맞추기
                aligned = np.zeros((len(te), len(classes)))
                for j, c in enumerate(pipe.classes_):
                    aligned[:, np.where(classes == c)[0][0]] = proba[:, j]
                proba_sum += aligned
                n_used += 1
            except Exception:  # noqa: BLE001
                continue
        if n_used == 0:
            continue
        pred = classes[np.argmax(proba_sum, axis=1)]
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.array(accs), np.array(f1s)


def run_experiment(df: pd.DataFrame, combo: str, fusion: str, clf_name: str,
                   skf: StratifiedKFold) -> Dict[str, object]:
    y = df["label"].to_numpy()
    blocks_cols = combo_feature_blocks(combo)

    if fusion == "late" and len(blocks_cols) >= 2:
        blocks = {m: df[cols].to_numpy(float) for m, cols in blocks_cols.items()}
        accs, f1s = _eval_late(blocks, y, clf_name, skf)
    elif fusion == "feature" and len(blocks_cols) >= 2:
        blocks = {m: df[cols].to_numpy(float) for m, cols in blocks_cols.items()}
        accs, f1s = _eval_feature(blocks, y, clf_name, skf)
    else:
        # early, 또는 단일 모달(융합 무의미 → concat 동일)
        cols = [c for cols in blocks_cols.values() for c in cols]
        X = df[cols].to_numpy(float)
        accs, f1s = _eval_concat(X, y, clf_name, skf)

    return {
        "combo": combo, "fusion": fusion, "classifier": clf_name,
        "n_features": sum(len(c) for c in blocks_cols.values()),
        "acc_mean": float(np.mean(accs)) if len(accs) else float("nan"),
        "acc_std": float(np.std(accs)) if len(accs) else float("nan"),
        "f1_mean": float(np.mean(f1s)) if len(f1s) else float("nan"),
        "f1_std": float(np.std(f1s)) if len(f1s) else float("nan"),
        "_accs": accs,
    }


# =============================================================================
# 전체 63 실험 + ablation
# =============================================================================

def run_all(df: pd.DataFrame, out_prefix: str) -> pd.DataFrame:
    n_total = len(df)
    df = df.dropna(subset=["label"]).copy()
    if len(df) == 0:
        raise ValueError(
            f"라벨(label)이 있는 trial이 0개입니다 (전체 {n_total}개). "
            "feature_table 의 obj_id/obj_name 이 9클래스와 매칭되지 않았습니다.\n"
            "  확인:  python -c \"import pandas as pd;"
            "df=pd.read_csv('<feature_table>.csv');"
            "print(df[['obj_id','obj_name']].drop_duplicates())\"\n"
            "  → 나온 값을 stage1_classes.py 의 이름 규칙과 맞추거나 알려주세요."
        )
    if len(df) < n_total:
        LOG.warning("라벨 없는 trial %d개 제외 (%d→%d)", n_total - len(df), n_total, len(df))
    df["label"] = df["label"].astype(int)
    y = df["label"].to_numpy()
    n_per = pd.Series(y).value_counts()
    n_splits = min(N_SPLITS, int(n_per.min()))
    if n_splits < 2:
        raise ValueError(f"클래스당 표본이 너무 적어 CV 불가 (min={n_per.min()})")
    if n_splits < N_SPLITS:
        LOG.warning("클래스당 최소 표본 %d → n_splits=%d 로 축소", n_per.min(), n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results, fold_acc = [], {}
    for combo, fusion, clf in itertools.product(MODAL_COMBOS, FUSIONS, CLASSIFIERS):
        r = run_experiment(df, combo, fusion, clf, skf)
        fold_acc[(combo, fusion, clf)] = r.pop("_accs")
        results.append(r)
        LOG.info("[%s | %-7s | %s] acc=%.3f±%.3f  f1=%.3f",
                 combo, fusion, clf, r["acc_mean"], r["acc_std"], r["f1_mean"])

    res_df = pd.DataFrame(results)
    res_path = Path(f"results_{out_prefix}.csv")
    res_df.to_csv(res_path, index=False)
    LOG.info("결과 저장: %s (%d 실험)", res_path, len(res_df))

    _ablation(fold_acc, out_prefix)
    _maybe_shap(df, out_prefix)
    _modality_diagnostics(df, out_prefix, skf)
    return res_df


def _modality_diagnostics(df: pd.DataFrame, out_prefix: str, skf: StratifiedKFold) -> None:
    """사후 진단 지표(피처 아님): 모달별 분류기의 out-of-fold 예측으로
      · modality_agreement_score : 모달(T/A/G)별 예측이 서로 일치하는 trial 비율
      · fused_confidence_entropy  : late-fusion 확률의 엔트로피(불확실성)
    v2b 에서 NaN 이던 두 크로스모달 지표를, '학습된 분류기 출력'이 생기는
    이 단계에서 계산해 trial 단위로 저장한다.
    출력: diagnostics_<prefix>.csv (trial 행) + 콘솔 요약."""
    y = df["label"].astype(int).to_numpy()
    classes = np.unique(y)
    mods = ["T", "A", "G"]
    # 모달별 OOF 예측/확률
    oof_pred = {m: np.full(len(y), -1) for m in mods}
    oof_prob = {m: np.zeros((len(y), len(classes))) for m in mods}
    have = {}
    for m in mods:
        cols = [c for c in FEATURE_GROUPS[m] if c in df.columns]
        if not cols:
            continue
        have[m] = True
        X = df[cols].to_numpy(float)
        for tr, te in skf.split(X, y):
            pipe = _pipe("RF")
            try:
                pipe.fit(X[tr], y[tr])
                pr = pipe.predict_proba(X[te])
                aligned = np.zeros((len(te), len(classes)))
                for j, cc in enumerate(pipe.classes_):
                    aligned[:, np.where(classes == cc)[0][0]] = pr[:, j]
                oof_prob[m][te] = aligned
                oof_pred[m][te] = classes[np.argmax(aligned, axis=1)]
            except Exception:  # noqa: BLE001
                continue
    used = [m for m in mods if have.get(m)]
    if len(used) < 2:
        LOG.info("모달 진단 생략(사용 가능 모달 < 2)")
        return
    # agreement: 사용 모달들의 예측이 최빈값과 일치하는 비율(모달 간 일치도)
    preds = np.vstack([oof_pred[m] for m in used]).T  # (n, n_mod)
    agree = np.array([
        np.max(np.bincount(row[row >= 0])) / max(1, np.sum(row >= 0))
        if np.any(row >= 0) else np.nan for row in preds])
    # fused prob = 사용 모달 확률 평균 → 엔트로피
    fused = np.mean([oof_prob[m] for m in used], axis=0)
    fused = fused / (fused.sum(axis=1, keepdims=True) + 1e-12)
    ent = -np.sum(fused * np.log(fused + 1e-12), axis=1)
    diag = pd.DataFrame({
        "label": y,
        "xm_modality_agreement_score": agree,
        "xm_fused_confidence_entropy": ent,
    })
    if "trial_dir" in df.columns:
        diag.insert(0, "trial_dir", df["trial_dir"].to_numpy())
    path = Path(f"diagnostics_{out_prefix}.csv")
    diag.to_csv(path, index=False)
    LOG.info("모달 진단 저장: %s  (agreement 평균 %.3f, entropy 평균 %.3f; 사용 모달 %s)",
             path, float(np.nanmean(agree)), float(np.mean(ent)), "+".join(used))


def _ablation(fold_acc: Dict, out_prefix: str) -> None:
    """Full(T+A+G) vs Full−G/−T/−A, late fusion 기준, fold별 paired t-test."""
    rows = []
    drop_map = {"-G": "T+A", "-T": "A+G", "-A": "T+G"}
    for clf in CLASSIFIERS:
        full_key = ("T+A+G", "late", clf)
        if full_key not in fold_acc:
            continue
        full = fold_acc[full_key]
        for tag, combo in drop_map.items():
            key = (combo, "late", clf)
            if key not in fold_acc:
                continue
            reduced = fold_acc[key]
            n = min(len(full), len(reduced))
            if n < 2:
                continue
            diff = full[:n] - reduced[:n]
            try:
                t, p = stats.ttest_rel(full[:n], reduced[:n])
            except Exception:  # noqa: BLE001
                t, p = float("nan"), float("nan")
            rows.append({
                "classifier": clf, "ablation": tag,
                "full_acc": float(np.mean(full)),
                "reduced_combo": combo, "reduced_acc": float(np.mean(reduced)),
                "delta_acc": float(np.mean(diff)),
                "t_stat": float(t), "p_value": float(p),
                "significant_0.05": bool(p < 0.05) if p == p else False,
            })
    ab = pd.DataFrame(rows)
    path = Path(f"ablation_{out_prefix}.csv")
    ab.to_csv(path, index=False)
    LOG.info("ablation 저장: %s", path)
    # 핵심 결과(−G) 강조 출력
    g = ab[ab["ablation"] == "-G"]
    if len(g):
        LOG.info("=== 후각 ablation (−G) 핵심 결과 ===")
        for _, r in g.iterrows():
            LOG.info("  %s: Full %.3f → T+A %.3f  (Δ=%.3f, p=%.4g, %s)",
                     r["classifier"], r["full_acc"], r["reduced_acc"], r["delta_acc"],
                     r["p_value"], "유의" if r["significant_0.05"] else "비유의")


def _maybe_shap(df: pd.DataFrame, out_prefix: str) -> None:
    try:
        import shap  # noqa: F401
    except ImportError:
        LOG.info("shap 미설치 — SHAP 분석 생략 (pip install shap 로 활성화)")
        return
    try:
        import shap
        cols = [c for cols in FEATURE_GROUPS.values() for c in cols]
        X = SimpleImputer(strategy="median").fit_transform(df[cols])
        y = df["label"].astype(int).to_numpy()
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        sv = shap.TreeExplainer(rf).shap_values(X)
        # shap 버전별 반환형 정규화 → 피처별 평균|SHAP|
        if isinstance(sv, list):
            imp = np.mean([np.abs(np.asarray(s)).mean(axis=0) for s in sv], axis=0)
        else:
            arr = np.asarray(sv)
            imp = np.abs(arr).mean(axis=(0, 2)) if arr.ndim == 3 else np.abs(arr).mean(axis=0)
        if len(imp) != len(cols):
            raise ValueError(f"SHAP 길이 불일치 {len(imp)} vs {len(cols)}")
        out = pd.DataFrame({"feature": cols, "mean_abs_shap": imp}) \
            .sort_values("mean_abs_shap", ascending=False)
        out.to_csv(f"shap_{out_prefix}.csv", index=False)
        LOG.info("SHAP 저장: shap_%s.csv (상위: %s)", out_prefix,
                 ", ".join(out["feature"].head(5)))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("SHAP 분석 실패: %s", exc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="63 실험 + ablation (Stratified 5-fold)")
    ap.add_argument("--features", required=True, help="feature_table.csv 경로")
    ap.add_argument("--out-prefix", default="stage1")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    df = pd.read_csv(args.features)
    run_all(df, args.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
