#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_bme_profile.py
=======================================================================
BME688 다중온도 스캔의 '온도 스텝별 정보량'을 파일럿 데이터로 진단하고,
쓸모없는(=저온 미도달로 신호 없는) 스텝을 자동 제외/병합할 목록을 만든다.

[왜 필요한가 — 정직한 근거]
소프트웨어 온도 스캔은 setpoint 를 명령할 뿐, 저온(100°C) 스텝이 열관성 때문에
실제 목표 온도에 도달했는지 보장하지 못한다(이전 논의). 그리고 BME688 은
Adafruit 라이브러리에서 '실제 플레이트 온도'를 노출하지 않는다(temp_c 는 주변
온도에 가깝다). 따라서 '온도가 맞았는지'를 직접 재는 대신, 조작적으로 의미 있는
질문을 데이터로 답한다:

  질문: "이 온도 스텝의 gas resistance 가 VOC 노출(sniff) vs 무노출(baseline)에서
        유의하게 다른가?  스텝 간 저항이 구분되는가?"
  → 저온 스텝이 목표 온도에 못 갔다면, 이웃 스텝과 저항이 사실상 같거나
    VOC 반응이 없어서 판별 기여가 0 에 가깝다. 그런 스텝을 걸러낸다.

[진단 지표] 각 setpoint 온도 T 에 대해:
  1. sep_vs_neighbor : 이웃 온도 스텝과 저항 median 차이의 상대크기
                       (작을수록 '온도가 안 갈라짐' = 미도달 의심)
  2. voc_response    : |baseline median − sniff median| / baseline median
                       (작을수록 'VOC 반응 없음')
  3. class_effect    : 클래스(label)별 sniff 저항의 분산비(between/within),
                       클수록 '이 온도가 클래스를 가른다'
  4. cv_across_trials: trial 간 변동계수(과도하면 불안정)

[자동 대응]
  --exclude-thresh 로 (voc_response < t) & (sep_vs_neighbor < t) 스텝을 '제외 후보'로.
  결과 bme_profile_steps.json 에 keep/exclude/merge 목록을 저장.
  extract_features_v2b_profile.py 는 이 파일이 있으면 읽어서 해당 온도 스텝을
  fingerprint 계산에서 제외/병합한다(BME_STEP_POLICY).

사용법:
  # 진단 리포트 + 정책 파일 생성
  python diagnose_bme_profile.py --data-root data/raw/<pilot_session> \
      --stage 2a --out-report bme_profile_report.csv --out-policy bme_profile_steps.json

  # 임계 조정
  python diagnose_bme_profile.py --data-root ... --exclude-thresh 0.02
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def _read_csv_auto(path: Path):
    if not path.exists():
        return None, False
    try:
        with open(path) as fh:
            first = fh.readline().strip()
    except Exception:
        return None, False
    try:
        float(first.split(",")[0]); is_header = False
    except (ValueError, IndexError):
        is_header = True
    try:
        return (pd.read_csv(path), True) if is_header else (pd.read_csv(path, header=None), False)
    except Exception:
        return None, False


def _events(trial: Path):
    f = trial / "events.csv"
    df, hh = _read_csv_auto(f)
    if df is None:
        return {}
    if hh:
        tcol = "t_us" if "t_us" in df.columns else df.columns[0]
        pcol = "phase_name" if "phase_name" in df.columns else df.columns[-1]
        df = df.rename(columns={tcol: "t_us", pcol: "phase"})[["t_us", "phase"]]
    else:
        df.columns = ["t_us", "phase"][:df.shape[1]]
    rows = df.to_dict("records")
    names = [str(r["phase"]).strip() for r in rows]
    seg = {}
    if any(n.endswith("_start") for n in names):
        st = {}
        for r in rows:
            n = str(r["phase"]).strip(); t = float(r["t_us"])
            if n.endswith("_start"): st[n[:-6]] = t
            elif n.endswith("_end") and n[:-4] in st: seg.setdefault(n[:-4], []).append((st[n[:-4]], t))
    else:
        for i, r in enumerate(rows):
            t0 = float(r["t_us"]); t1 = float(rows[i+1]["t_us"]) if i+1 < len(rows) else np.inf
            seg.setdefault(str(r["phase"]).strip(), []).append((t0, t1))
    return seg


def _mask(t_us, seg, phase):
    m = np.zeros(len(t_us), bool)
    for t0, t1 in seg.get(phase, []):
        m |= (t_us >= t0) & (t_us < t1)
    return m


def collect(data_root: Path, resolver=None):
    """모든 trial 에서 (label, 온도별 baseline/sniff 저항)을 모은다."""
    trials = sorted(p for p in data_root.glob("**/trial_*") if p.is_dir())
    # per temp: baseline vals, sniff vals, per-trial sniff median (+label)
    base_by_T = defaultdict(list)
    sniff_by_T = defaultdict(list)
    trial_med = defaultdict(list)   # (T) -> list of (label, per-trial sniff median)
    for tr in trials:
        f = tr / "gas_bme688.csv"
        df, hh = _read_csv_auto(f)
        if df is None or not hh or "set_temp_c" not in df.columns:
            continue
        seg = _events(tr)
        t_us = df["t_us"].to_numpy(float)
        setT = df["set_temp_c"].to_numpy(float)
        R = df["gas_resistance_ohm"].to_numpy(float)
        mb = _mask(t_us, seg, "baseline"); ms = _mask(t_us, seg, "sniff")
        if ms.sum() < 3:
            ms = np.ones(len(t_us), bool)
        # label
        label = None
        mj = tr / "meta.json"
        if mj.exists():
            try:
                j = json.loads(mj.read_text()); label = j.get("label")
                if label is None and resolver is not None:
                    k = resolver.resolve_fuzzy(j.get("obj_id"), j.get("obj_name"), tr.parent.name)
                    label = k.label if k else None
            except Exception:
                pass
        for T in np.unique(setT):
            bvals = R[mb & (np.abs(setT - T) < 1)]
            svals = R[ms & (np.abs(setT - T) < 1)]
            if len(bvals): base_by_T[float(T)].extend(bvals.tolist())
            if len(svals):
                sniff_by_T[float(T)].extend(svals.tolist())
                trial_med[float(T)].append((label, float(np.median(svals))))
    return base_by_T, sniff_by_T, trial_med


def analyze(base_by_T, sniff_by_T, trial_med):
    temps = sorted(sniff_by_T.keys())
    sniff_med = {T: float(np.median(sniff_by_T[T])) for T in temps}
    rows = []
    for i, T in enumerate(temps):
        base_med = float(np.median(base_by_T[T])) if base_by_T.get(T) else np.nan
        s_med = sniff_med[T]
        voc = abs(base_med - s_med) / (abs(base_med) + 1e-9) if np.isfinite(base_med) else np.nan
        # 이웃과의 분리도(상대)
        neigh = []
        if i > 0: neigh.append(sniff_med[temps[i-1]])
        if i < len(temps)-1: neigh.append(sniff_med[temps[i+1]])
        sep = np.mean([abs(s_med - n) / (abs(s_med) + 1e-9) for n in neigh]) if neigh else np.nan
        # 클래스 효과(between/within 분산비)
        lab_vals = defaultdict(list)
        for lab, v in trial_med[T]:
            if lab is not None: lab_vals[lab].append(v)
        cls_eff = np.nan
        if len(lab_vals) >= 2:
            allv = [v for vs in lab_vals.values() for v in vs]
            grand = np.mean(allv)
            between = np.mean([len(vs)*(np.mean(vs)-grand)**2 for vs in lab_vals.values()])
            within = np.mean([np.var(vs) for vs in lab_vals.values() if len(vs) >= 2]) + 1e-9
            cls_eff = float(between / within)
        # trial 간 변동계수
        pv = [v for _, v in trial_med[T]]
        cv = float(np.std(pv) / (np.mean(pv) + 1e-9)) if len(pv) >= 2 else np.nan
        rows.append({"set_temp_c": T, "n_sniff": len(sniff_by_T[T]),
                     "baseline_med": base_med, "sniff_med": s_med,
                     "voc_response": voc, "sep_vs_neighbor": sep,
                     "class_effect": cls_eff, "cv_across_trials": cv})
    return pd.DataFrame(rows)


def decide(report: pd.DataFrame, exclude_thresh: float):
    """제외/병합 정책 산출.
      제외 후보: voc_response < t AND sep_vs_neighbor < t (신호 없음 + 온도 안 갈라짐)
      병합: 연속 제외 후보들은 인접 keep 스텝으로 병합 표시(참고용)."""
    keep, exclude = [], []
    for _, r in report.iterrows():
        weak = (np.nan_to_num(r["voc_response"], nan=0) < exclude_thresh) and \
               (np.nan_to_num(r["sep_vs_neighbor"], nan=0) < exclude_thresh)
        (exclude if weak else keep).append(float(r["set_temp_c"]))
    return {"keep_temps": keep, "exclude_temps": exclude,
            "exclude_thresh": exclude_thresh,
            "note": "extract_features_v2b_profile.py 가 이 파일을 읽으면 exclude_temps 를 "
                    "fingerprint 계산에서 제외한다. 저온 미도달 스텝 정리용."}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="파일럿 세션 폴더")
    ap.add_argument("--stage", choices=["2a", "2b"], default=None)
    ap.add_argument("--exclude-thresh", type=float, default=0.01,
                    help="voc_response·sep_vs_neighbor 둘 다 이 값 미만이면 제외 후보(기본 0.01=1%%)")
    ap.add_argument("--out-report", default="bme_profile_report.csv")
    ap.add_argument("--out-policy", default="bme_profile_steps.json")
    a = ap.parse_args(argv)

    resolver = None
    if a.stage == "2a":
        import stage2a_classes as resolver
    elif a.stage == "2b":
        import stage2b_classes as resolver

    base_by_T, sniff_by_T, trial_med = collect(Path(a.data_root), resolver)
    if not sniff_by_T:
        print("[경고] 다중온도 BME688 데이터를 찾지 못함(set_temp_c 컬럼 필요). "
              "펌웨어가 다중온도 스캔인지 확인.")
        return
    report = analyze(base_by_T, sniff_by_T, trial_med)
    report.to_csv(a.out_report, index=False)
    policy = decide(report, a.exclude_thresh)
    Path(a.out_policy).write_text(json.dumps(policy, ensure_ascii=False, indent=2))

    print("\n=== BME688 온도 스텝 진단 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
        print(report.to_string(index=False))
    print(f"\nkeep  ({len(policy['keep_temps'])}): {policy['keep_temps']}")
    print(f"exclude({len(policy['exclude_temps'])}): {policy['exclude_temps']}  "
          f"(thresh={a.exclude_thresh})")
    print(f"\n리포트: {a.out_report}\n정책  : {a.out_policy}")
    if policy["exclude_temps"]:
        print("→ 이 정책 파일을 추출기와 같은 폴더에 두면, 해당 온도 스텝이 "
              "fingerprint 에서 자동 제외됩니다.")
    else:
        print("→ 제외 대상 없음: 모든 온도 스텝이 정보량을 가짐(저온 포함 OK).")


if __name__ == "__main__":
    main()
