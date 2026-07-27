#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ml_strict.py
Final strict-modality ML runner for redesigned Stage1/Stage2 feature tables.

Key rule:
  T+A   -> T + A + TA only
  T+O   -> T + O + TO only
  A+O   -> A + O + AO only
  Full  -> T + A + O + TA + TO + AO
This prevents acoustic-derived cross features from leaking into T+O.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
N_SPLITS = 5
COMBOS = ["T", "A", "O", "T+A", "T+O", "A+O", "T+A+O"]
FUSIONS = ["early", "feature", "late"]
CLFS = ["RF", "SVM", "MLP"]

LOG = logging.getLogger("run_ml_strict")


def default_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {}
    for g in ["T", "A", "O", "TA", "TO", "AO"]:
        groups[g] = [c for c in df.columns if c.startswith(g + "_")]
    return groups


def load_groups(path: str | None, df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = default_groups(df)
    if path:
        p = Path(path)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            for g in ["T", "A", "O", "TA", "TO", "AO"]:
                if g in raw:
                    groups[g] = [c for c in raw[g] if c in df.columns]
    return groups


def combo_groups(combo: str) -> List[str]:
    return {
        "T": ["T"],
        "A": ["A"],
        "O": ["O"],
        "T+A": ["T", "A", "TA"],
        "T+O": ["T", "O", "TO"],
        "A+O": ["A", "O", "AO"],
        "T+A+O": ["T", "A", "O", "TA", "TO", "AO"],
    }[combo]


def combo_columns(combo: str, groups: Dict[str, List[str]]) -> List[str]:
    cols: List[str] = []
    for g in combo_groups(combo):
        cols += groups.get(g, [])
    # preserve order, remove duplicates
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def make_classifier(name: str):
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                             random_state=RANDOM_STATE, early_stopping=True)
    raise ValueError(name)


def pipe(name: str) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", make_classifier(name)),
    ])


def eval_concat(X: np.ndarray, y: np.ndarray, clf_name: str, skf) -> Tuple[np.ndarray, np.ndarray]:
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        p = pipe(clf_name)
        p.fit(X[tr], y[tr])
        pred = p.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.asarray(accs), np.asarray(f1s)


def eval_feature(block_arrays: Dict[str, np.ndarray], y: np.ndarray, clf_name: str, skf) -> Tuple[np.ndarray, np.ndarray]:
    accs, f1s = [], []
    first = next(iter(block_arrays.values()))
    for tr, te in skf.split(first, y):
        Xtr_parts, Xte_parts = [], []
        for Xb in block_arrays.values():
            imp = SimpleImputer(strategy="median")
            sc = StandardScaler()
            btr = sc.fit_transform(imp.fit_transform(Xb[tr]))
            bte = sc.transform(imp.transform(Xb[te]))
            ncomp = max(1, min(btr.shape[1], btr.shape[0] - 1))
            pca = PCA(n_components=ncomp, random_state=RANDOM_STATE)
            Xtr_parts.append(pca.fit_transform(btr))
            Xte_parts.append(pca.transform(bte))
        Xtr, Xte = np.hstack(Xtr_parts), np.hstack(Xte_parts)
        clf = make_classifier(clf_name)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.asarray(accs), np.asarray(f1s)


def eval_late(block_arrays: Dict[str, np.ndarray], y: np.ndarray, clf_name: str, skf) -> Tuple[np.ndarray, np.ndarray]:
    accs, f1s = [], []
    classes = np.unique(y)
    first = next(iter(block_arrays.values()))
    for tr, te in skf.split(first, y):
        proba_sum = np.zeros((len(te), len(classes)))
        used = 0
        for _, Xb in block_arrays.items():
            if Xb.shape[1] == 0:
                continue
            p = pipe(clf_name)
            p.fit(Xb[tr], y[tr])
            proba = p.predict_proba(Xb[te])
            aligned = np.zeros((len(te), len(classes)))
            for j, c in enumerate(p.classes_):
                aligned[:, np.where(classes == c)[0][0]] = proba[:, j]
            proba_sum += aligned
            used += 1
        if used == 0:
            continue
        pred = classes[np.argmax(proba_sum, axis=1)]
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.asarray(accs), np.asarray(f1s)


def run_one(df: pd.DataFrame, groups: Dict[str, List[str]], combo: str, fusion: str, clf_name: str, skf):
    y = df["label"].astype(int).to_numpy()
    gnames = combo_groups(combo)
    block_cols = {g: [c for c in groups.get(g, []) if c in df.columns] for g in gnames}
    block_cols = {g: c for g, c in block_cols.items() if len(c) > 0}
    cols = [c for cs in block_cols.values() for c in cs]
    if not cols:
        raise ValueError(f"No columns for combo={combo}")

    if fusion == "early" or len(block_cols) == 1:
        X = df[cols].to_numpy(float)
        accs, f1s = eval_concat(X, y, clf_name, skf)
    else:
        Xb = {g: df[c].to_numpy(float) for g, c in block_cols.items()}
        if fusion == "feature":
            accs, f1s = eval_feature(Xb, y, clf_name, skf)
        elif fusion == "late":
            accs, f1s = eval_late(Xb, y, clf_name, skf)
        else:
            raise ValueError(fusion)
    return {
        "combo": combo, "fusion": fusion, "classifier": clf_name,
        "n_features": len(cols),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "_accs": accs,
    }


def clean_df(df: pd.DataFrame, label_col: str, drop_qc_bad: bool) -> pd.DataFrame:
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    df = df.dropna(subset=["label"]).copy()
    if drop_qc_bad:
        qc_cols = [c for c in ["QC_ok_tactile", "QC_ok_acoustic", "QC_ok_olfactory"] if c in df.columns]
        for c in qc_cols:
            df = df[pd.to_numeric(df[c], errors="coerce").fillna(0).astype(bool)]
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def run_all(df: pd.DataFrame, groups: Dict[str, List[str]], out_prefix: str):
    y = df["label"].astype(int).to_numpy()
    n_per = pd.Series(y).value_counts()
    n_splits = min(N_SPLITS, int(n_per.min()))
    if n_splits < 2:
        raise ValueError(f"CV 불가: 클래스당 최소 표본이 너무 적음 min={n_per.min()}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows, fold_acc = [], {}
    for combo, fusion, clf in itertools.product(COMBOS, FUSIONS, CLFS):
        r = run_one(df, groups, combo, fusion, clf, skf)
        fold_acc[(combo, fusion, clf)] = r.pop("_accs")
        rows.append(r)
        LOG.info("[%s | %-7s | %s] acc=%.3f±%.3f f1=%.3f", combo, fusion, clf, r["acc_mean"], r["acc_std"], r["f1_mean"])
    res = pd.DataFrame(rows)
    res.to_csv(f"results_{out_prefix}.csv", index=False)

    best = res.sort_values(["combo", "acc_mean", "f1_mean"], ascending=[True, False, False]).groupby("combo", as_index=False).head(1)
    best.to_csv(f"best_per_combo_{out_prefix}.csv", index=False)

    ab_rows = []
    for clf in CLFS:
        full_key = ("T+A+O", "late", clf)
        if full_key not in fold_acc:
            continue
        full = fold_acc[full_key]
        for tag, reduced_combo in [("-O", "T+A"), ("-T", "A+O"), ("-A", "T+O")]:
            key = (reduced_combo, "late", clf)
            reduced = fold_acc[key]
            n = min(len(full), len(reduced))
            t, p = stats.ttest_rel(full[:n], reduced[:n]) if n >= 2 else (np.nan, np.nan)
            ab_rows.append({
                "classifier": clf, "ablation": tag,
                "full_combo": "T+A+O", "full_acc": float(np.mean(full)),
                "reduced_combo": reduced_combo, "reduced_acc": float(np.mean(reduced)),
                "delta_acc": float(np.mean(full[:n] - reduced[:n])),
                "t_stat": float(t), "p_value": float(p),
                "significant_0.05": bool(p < 0.05) if np.isfinite(p) else False,
            })
    pd.DataFrame(ab_rows).to_csv(f"ablation_{out_prefix}.csv", index=False)
    return res, best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Strict no-leakage ML for redesigned feature tables")
    ap.add_argument("--features", required=True)
    ap.add_argument("--groups-json", default=None)
    ap.add_argument("--out-prefix", default="strict")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--keep-qc-bad", action="store_true", help="Do not drop rows failing QC flags")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    df0 = pd.read_csv(args.features)
    groups = load_groups(args.groups_json, df0)
    df = clean_df(df0, args.label_col, drop_qc_bad=not args.keep_qc_bad)
    LOG.info("samples=%d classes=%d", len(df), df["label"].nunique())
    LOG.info("groups=%s", {k: len(v) for k, v in groups.items()})
    run_all(df, groups, args.out_prefix)
    LOG.info("saved: results_%s.csv / best_per_combo_%s.csv / ablation_%s.csv", args.out_prefix, args.out_prefix, args.out_prefix)


if __name__ == "__main__":
    main()
