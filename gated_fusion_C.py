"""
gated_fusion_C.py
Tuned Hybrid Gated Fusion (C 모델) — 기존 gated_fusion.py 업그레이드
======================================================================

포지셔닝 (근거: C_Model_Tuned_Hybrid_Gated_Fusion_Final.docx §1, §7)
--------------------------------------------------------------------
- 최고 "정확도" 모델은 early-RF (run_ml.py). 이 스크립트는 정확도 경쟁자가 아니라
  "sample별 모달 신뢰도(w_T,w_A,w_G) + 불확실성 진단(entropy, consistency)"을
  산출하는 confidence-aware neural fusion 후보다.
- 소표본(270 trial, 9클래스)에서는 RF가 이길 가능성이 크다(문서 §7). 따라서
  이 모델의 산출물은 정확도 그 자체보다 gate weight / 진단이 핵심이다.

구조 (근거: C 문서 §3~§5)
-------------------------
  z_m = Encoder_m(x_m),  m in {T, A, G}
  h   = HybridMLP([z_T, z_A, z_G, (x_X)])         # early-style 상호작용 학습
  w   = softmax(Gate([z_T, z_A, z_G, h]))         # sample별 모달 신뢰도, 합=1
  z_w = [w_T z_T, w_A z_A, w_G z_G]
  y_hat = Classifier([h, z_w])                    # 9-class
  L = L_joint + lam_aux * mean_m L_m + lam_reg * R(gate)

검증 (근거: C 문서 §2 "nested tuning")
--------------------------------------
- outer: Stratified 5-fold (Stage 1 관례, ML_설계_정리.md §4)
- inner: outer-train 내부 Stratified 3-fold 로 HP 선택 → optimistic bias 완화
- 최종 refit 은 outer-train 전체(early-stop용 15% val 분리) 후 outer-test 평가

[구현 선택] 표시
----------------
- 인코더/hybrid/gate 를 소형 MLP 로 고정(과적합 방지). 문서에 구체 층 수 미명시.
- gate 정규화 R = -entropy(w) 의 음수(=엔트로피 패널티 완화용 약한 항). lam_reg 작게.
- --factorized 는 C+ 문서(§3 factorized heads)의 요소 중 "현재 데이터로 공짜"인
  material/content head 만 앞당긴 옵션. 나머지 C+ 요소(action/reliability/active/
  physics)는 Stage 1 데이터로 검증 불가하여 미구현.

주의 (정보 부족 → 명시)
----------------------
- FEATURE_GROUPS 는 extract_features 에서 import (컬럼명 추측 안 함).
- label→(material,content) 매핑은 label 이 재질-major(glass0/1/2, ceramic3/4/5,
  plastic6/7/8)라는 "가정"이다. stage1_classes.py 순서가 다르면 --content-first 사용.
  확실치 않으면 --factorized 없이 먼저 돌릴 것.

사용법
------
  python gated_fusion_C.py --features feature_table_stage1.csv
  python gated_fusion_C.py --features f.csv --factorized      # C+ head 옵션
  python gated_fusion_C.py --features f.csv --label-col label --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("C-THGF")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# run_ml.py 와 동일한 피처 정의를 그대로 사용 (컬럼명 추측 금지)
try:
    from extract_features import FEATURE_GROUPS
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        "extract_features.FEATURE_GROUPS 를 import 하지 못했습니다.\n"
        "이 스크립트는 run_ml.py 와 같은 폴더(같은 프로젝트)에서 실행해야 합니다.\n"
        f"원인: {e}"
    )


# =============================================================================
# 모달 블록 추출
# =============================================================================

def modality_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    """FEATURE_GROUPS 기준 T/A/G 블록 + (있으면) X 크로스모달 블록.
    feature_table 에 실제 존재하는 컬럼만 사용한다."""
    blocks: Dict[str, List[str]] = {}
    for m in ("T", "A", "G"):
        cols = [c for c in FEATURE_GROUPS[m] if c in df.columns]
        if not cols:
            raise ValueError(f"모달 {m} 의 피처 컬럼이 feature_table 에 없습니다.")
        blocks[m] = cols
    if "X" in FEATURE_GROUPS:
        xcols = [c for c in FEATURE_GROUPS["X"] if c in df.columns]
        if xcols:
            blocks["X"] = xcols  # hybrid 브랜치에만 투입 (게이트 대상 아님)
    return blocks


# =============================================================================
# 전처리 (fold 별 train 통계로 fit — leakage 방지)
# =============================================================================

class BlockPrep:
    """모달 블록별 median-impute + standardize. train 에서만 fit."""

    def __init__(self, blocks: Dict[str, List[str]]):
        self.blocks = blocks
        self.imp: Dict[str, SimpleImputer] = {}
        self.sc: Dict[str, StandardScaler] = {}

    def fit(self, df: pd.DataFrame) -> "BlockPrep":
        for m, cols in self.blocks.items():
            X = df[cols].to_numpy(float)
            self.imp[m] = SimpleImputer(strategy="median").fit(X)
            self.sc[m] = StandardScaler().fit(self.imp[m].transform(X))
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        out = {}
        for m, cols in self.blocks.items():
            X = df[cols].to_numpy(float)
            X = self.sc[m].transform(self.imp[m].transform(X))
            out[m] = torch.tensor(X, dtype=torch.float32)
        return out


# =============================================================================
# 모델 (C 문서 §3~§5)
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, z_dim), nn.ReLU(),
        )

    def forward(self, x): return self.net(x)


class TunedHybridGatedFusion(nn.Module):
    """C 모델. T/A/G 인코더 + early-style hybrid 브랜치 + attention gate."""

    def __init__(self, dims: Dict[str, int], n_class: int,
                 hidden: int = 32, z_dim: int = 16,
                 factorized: bool = False, n_material: int = 3, n_content: int = 3):
        super().__init__()
        self.factorized = factorized
        self.enc = nn.ModuleDict({
            m: Encoder(dims[m], hidden, z_dim) for m in ("T", "A", "G")
        })
        x_extra = dims.get("X", 0)  # 크로스모달 피처는 hybrid 브랜치에만 concat
        self.hybrid = nn.Sequential(
            nn.Linear(3 * z_dim + x_extra, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, z_dim), nn.ReLU(),
        )
        # gate: [z_T, z_A, z_G, h] -> 3 (softmax, 합=1)
        self.gate = nn.Sequential(
            nn.Linear(4 * z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        # 최종 분류기: [h, w_T z_T, w_A z_A, w_G z_G]
        self.classifier = nn.Sequential(
            nn.Linear(z_dim + 3 * z_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_class),
        )
        # 모달별 aux head (각 인코더가 독립적으로도 class 정보 유지 — C §3)
        self.aux = nn.ModuleDict({
            m: nn.Linear(z_dim, n_class) for m in ("T", "A", "G")
        })
        # [C+ 차용, 선택] material/content factorized head (C+ §3)
        if factorized:
            self.head_mat = nn.Linear(z_dim, n_material)
            self.head_con = nn.Linear(z_dim, n_content)

    def forward(self, xb: Dict[str, torch.Tensor]):
        z = {m: self.enc[m](xb[m]) for m in ("T", "A", "G")}
        hyb_in = [z["T"], z["A"], z["G"]]
        if "X" in xb:
            hyb_in.append(xb["X"])
        h = self.hybrid(torch.cat(hyb_in, dim=1))

        w = F.softmax(self.gate(torch.cat([z["T"], z["A"], z["G"], h], dim=1)), dim=1)
        z_w = torch.cat([w[:, 0:1] * z["T"],
                         w[:, 1:2] * z["A"],
                         w[:, 2:3] * z["G"]], dim=1)
        logits = self.classifier(torch.cat([h, z_w], dim=1))
        aux = {m: self.aux[m](z[m]) for m in ("T", "A", "G")}

        out = {"logits": logits, "w": w, "aux": aux, "h": h}
        if self.factorized:
            out["mat"] = self.head_mat(h)
            out["con"] = self.head_con(h)
        return out


# =============================================================================
# 학습 / 평가
# =============================================================================

def _to_dev(xb): return {m: t.to(DEVICE) for m, t in xb.items()}


def train_model(model, Xtr, ytr, Xval, yval, *, hp, epochs, patience,
                mat_tr=None, con_tr=None) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    Xtr, Xval = _to_dev(Xtr), _to_dev(Xval)
    ytr = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    yval = torch.tensor(yval, dtype=torch.long, device=DEVICE)
    if model.factorized:
        mat_tr = torch.tensor(mat_tr, dtype=torch.long, device=DEVICE)
        con_tr = torch.tensor(con_tr, dtype=torch.long, device=DEVICE)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr)
        loss = F.cross_entropy(out["logits"], ytr)                       # L_joint
        loss = loss + hp["lam_aux"] * torch.stack(                        # + aux
            [F.cross_entropy(out["aux"][m], ytr) for m in ("T", "A", "G")]
        ).mean()
        # R(gate): 평균 gate 가 한 모달로 붕괴하지 않도록 약한 엔트로피 보상
        w_mean = out["w"].mean(0)
        gate_ent = -(w_mean * (w_mean + 1e-9).log()).sum()
        loss = loss - hp["lam_reg"] * gate_ent
        if model.factorized:
            loss = loss + hp["lam_fac"] * (
                F.cross_entropy(out["mat"], mat_tr) + F.cross_entropy(out["con"], con_tr)
            )
        loss.backward()
        opt.step()

        # early stopping on val macro-F1
        model.eval()
        with torch.no_grad():
            pv = model(Xval)["logits"].argmax(1).cpu().numpy()
        f1 = f1_score(yval.cpu().numpy(), pv, average="macro")
        if f1 > best_f1:
            best_f1, best_state, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def evaluate(model, Xte, yte) -> Dict[str, object]:
    model.eval()
    out = model(_to_dev(Xte))
    prob = F.softmax(out["logits"], dim=1).cpu().numpy()
    pred = prob.argmax(1)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    # 진단 (C §2, §3): fused entropy / modality consistency / gate weight
    ent = -(prob * np.log(prob + 1e-9)).sum(1)                    # per-sample
    aux_pred = np.stack([out["aux"][m].argmax(1).cpu().numpy()
                         for m in ("T", "A", "G")], axis=1)       # (N,3)
    # consistency = 세 모달 aux head 다수결이 최종 예측과 일치하는 비율
    from scipy.stats import mode
    aux_major = mode(aux_pred, axis=1, keepdims=False).mode
    consistency = float((aux_major == pred).mean())
    w = out["w"].cpu().numpy()                                    # (N,3)
    return {
        "acc": acc, "f1": f1,
        "w_mean": w.mean(0),               # [w_T, w_A, w_G]
        "entropy_mean": float(ent.mean()),
        "consistency": consistency,
        "w_per_sample": w, "y": yte, "pred": pred,
    }


# =============================================================================
# Nested CV (C §2)
# =============================================================================

HP_GRID = [
    {"lr": 3e-3, "wd": 1e-3, "lam_aux": 0.3, "lam_reg": 0.01, "lam_fac": 0.3},
    {"lr": 1e-3, "wd": 1e-3, "lam_aux": 0.5, "lam_reg": 0.02, "lam_fac": 0.5},
    {"lr": 3e-3, "wd": 5e-4, "lam_aux": 0.2, "lam_reg": 0.005, "lam_fac": 0.2},
]


def _build(dims, n_class, factorized, n_mat, n_con):
    return TunedHybridGatedFusion(dims, n_class, factorized=factorized,
                                  n_material=n_mat, n_content=n_con).to(DEVICE)


def nested_cv(df, blocks, y, *, factorized, mat=None, con=None,
              epochs=200, patience=25) -> pd.DataFrame:
    dims = {m: len(c) for m, c in blocks.items()}
    n_class = int(len(np.unique(y)))
    n_mat = int(len(np.unique(mat))) if factorized else 3
    n_con = int(len(np.unique(con))) if factorized else 3

    outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for k, (tr_idx, te_idx) in enumerate(outer.split(df, y), 1):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ---- inner HP 선택 (optimistic bias 완화) ----
        inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
        hp_scores = []
        for hp in HP_GRID:
            f1s = []
            for i_tr, i_va in inner.split(df_tr, y_tr):
                prep = BlockPrep(blocks).fit(df_tr.iloc[i_tr])
                Xi_tr, Xi_va = prep.transform(df_tr.iloc[i_tr]), prep.transform(df_tr.iloc[i_va])
                m = _build(dims, n_class, factorized, n_mat, n_con)
                kw = {}
                if factorized:
                    kw = {"mat_tr": mat[tr_idx][i_tr], "con_tr": con[tr_idx][i_tr]}
                train_model(m, Xi_tr, y_tr[i_tr], Xi_va, y_tr[i_va],
                            hp=hp, epochs=epochs, patience=patience, **kw)
                r = evaluate(m, Xi_va, y_tr[i_va])
                f1s.append(r["f1"])
            hp_scores.append(np.mean(f1s))
        best_hp = HP_GRID[int(np.argmax(hp_scores))]

        # ---- outer-train 전체로 refit (early-stop용 15% val 분리) ----
        va_split = StratifiedKFold(6, shuffle=True, random_state=RANDOM_STATE)
        rt, rv = next(va_split.split(df_tr, y_tr))
        prep = BlockPrep(blocks).fit(df_tr.iloc[rt])
        Xr_tr, Xr_va = prep.transform(df_tr.iloc[rt]), prep.transform(df_tr.iloc[rv])
        Xr_te = prep.transform(df_te)
        model = _build(dims, n_class, factorized, n_mat, n_con)
        kw = {}
        if factorized:
            kw = {"mat_tr": mat[tr_idx][rt], "con_tr": con[tr_idx][rt]}
        train_model(model, Xr_tr, y_tr[rt], Xr_va, y_tr[rv],
                    hp=best_hp, epochs=epochs, patience=patience, **kw)
        res = evaluate(model, Xr_te, y_te)

        LOG.info("[fold %d] acc=%.3f f1=%.3f | w(T,A,G)=%.3f/%.3f/%.3f "
                 "ent=%.3f consist=%.3f | hp=%s",
                 k, res["acc"], res["f1"], *res["w_mean"],
                 res["entropy_mean"], res["consistency"], best_hp)
        rows.append({
            "fold": k, "acc": res["acc"], "f1": res["f1"],
            "w_T": res["w_mean"][0], "w_A": res["w_mean"][1], "w_G": res["w_mean"][2],
            "entropy": res["entropy_mean"], "consistency": res["consistency"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

def derive_factors(y: np.ndarray, content_first: bool) -> Tuple[np.ndarray, np.ndarray]:
    """[가정] 9클래스 factorial 을 재질/내용물 축으로 분해.
    기본: label 이 재질-major (glass0/1/2, ceramic3/4/5, plastic6/7/8).
    --content-first 지정 시 내용물-major 로 해석."""
    if content_first:
        material = y % 3
        content = y // 3
    else:
        material = y // 3
        content = y % 3
    return material.astype(int), content.astype(int)


def main():
    ap = argparse.ArgumentParser(description="Tuned Hybrid Gated Fusion (C 모델)")
    ap.add_argument("--features", required=True, help="feature_table CSV 경로")
    ap.add_argument("--label-col", default="label", help="정수 라벨 컬럼명(0..8)")
    ap.add_argument("--factorized", action="store_true",
                    help="[C+ 차용] material/content head 추가")
    ap.add_argument("--content-first", action="store_true",
                    help="label 이 내용물-major 순서일 때 지정")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out-prefix", default="C_THGF")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    if args.label_col not in df.columns:
        raise SystemExit(f"라벨 컬럼 '{args.label_col}' 이 없습니다. --label-col 로 지정하세요.")
    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    y = df[args.label_col].astype(int).to_numpy()

    blocks = modality_blocks(df)
    LOG.info("모달 블록: %s", {m: len(c) for m, c in blocks.items()})
    LOG.info("샘플 %d, 클래스 %d, device=%s", len(df), len(np.unique(y)), DEVICE)

    mat = con = None
    if args.factorized:
        mat, con = derive_factors(y, args.content_first)
        LOG.warning("[가정] material/content 분해 사용 — label 순서 확인 필수 "
                    "(다르면 --content-first). material=%d, content=%d 클래스",
                    len(np.unique(mat)), len(np.unique(con)))

    res = nested_cv(df, blocks, y, factorized=args.factorized, mat=mat, con=con,
                    epochs=args.epochs)

    LOG.info("\n===== 요약 (outer 5-fold) =====")
    LOG.info("accuracy   : %.3f ± %.3f", res["acc"].mean(), res["acc"].std())
    LOG.info("macro F1   : %.3f ± %.3f", res["f1"].mean(), res["f1"].std())
    LOG.info("gate weight: T=%.3f A=%.3f G=%.3f",
             res["w_T"].mean(), res["w_A"].mean(), res["w_G"].mean())
    LOG.info("entropy    : %.3f | consistency : %.3f",
             res["entropy"].mean(), res["consistency"].mean())
    LOG.info("※ 정확도는 early-RF 와 비교용. 이 모델의 주 산출물은 gate/진단이다 "
             "(C 문서 §1,§7).")

    out = Path(f"results_{args.out_prefix}.csv")
    res.to_csv(out, index=False)
    LOG.info("저장: %s", out)


if __name__ == "__main__":
    main()
