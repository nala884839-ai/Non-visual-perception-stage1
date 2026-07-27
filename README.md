# Stage 1 파이프라인 — 수집 → 피처추출 → ML/Ablation

ML_설계_정리.md **개정판**(2026-06-30)을 그대로 구현한 실행 파이프라인입니다.
하드웨어 수집 이후의 빠져 있던 단계(피처 추출 · 63 실험 · ablation)를 채웁니다.

## 반영된 확정 사항
- 촉각: SingleTact **s1 채널 고장 → s2/s3 2채널만** (force = s2 + s3), 전 trial 동일 기준
- 음향: Knowles BU **5kHz** 버스트(펌웨어 `BU_BURST_US=200` 확인), 접촉 구간 버스트 단위 추출
- BME688: **단일 온도 320°C**(`setGasHeater(320,150)` 확인) → 히터 스텝 개념 폐기,
  단일스텝 저항 곡선 통계량 **12피처로 재정의**
- 검증: **Stratified 5-fold CV 만** (LODO·날짜분산은 Stage 2)
- 수집: **단일 세션** 9클래스 × 30 trial = 270
- **수집 variant = `tap`** (Stage 1 목표는 후각 입증 → press/rub 없이 tap 접촉만으로 T/A 확보)
- 크로스모달: 계산 가능한 **4개만** 피처로 사용.
  `modality_consistency`·`fused_confidence_entropy`는 학습된 분류기 출력이 필요한
  **사후 진단 지표**라 피처에서 제외(결정 2026-06-30) → 필요 시 ML 단계에서 별도 계산.
- **액션(tap/press/rub) ablation은 미진행** — 모달 ablation(−G/−T/−A)만 수행.

## 파일
| 파일 | 역할 |
|---|---|
| `stage1_classes.py` | 9클래스(재질×내용물) ↔ obj_id/label 공용 매핑 |
| `collect_stage1.py` | ① 단일 세션 9클래스 수집 (기존 `TrialRunner` 호출, VOC 60s+ 간격) |
| `extract_features.py` | ② raw CSV → `feature_table.csv` (모달별 피처 + 메타/라벨) |
| `run_ml.py` | ③④ 63 실험(7조합×3융합×3분류기) + ablation paired t-test + (옵션)SHAP |
| `_smoke_test.py` | 하드웨어 없이 합성 데이터로 전체 파이프라인 검증 |

`collect_stage1.py`·`stage1_classes.py`는 프로젝트 루트(config.py·trial_runner.py 옆)에 두고 실행합니다.

## 실행 순서 (현재: 세션 = stage1_run1, tap-only)
```bash
# ① 수집은 이미 완료됨 → data/raw/stage1_run1/ 에 raw CSV 존재
#    (신규 수집 시:  python collect_stage1.py --variant tap --trials 30 --session stage1_run1)

# ② 피처 추출
python extract_features.py --session stage1_run1 --out feature_table_run1.csv

# ③④ ML + 모달 ablation (Stratified 5-fold)
python run_ml.py --features feature_table_run1.csv --out-prefix run1
#    → results_run1.csv   (63 실험 accuracy/F1)
#    → ablation_run1.csv  (Full vs Full−G/−T/−A, paired t-test) ← 후각 입증 핵심
```

## 하드웨어 없이 검증
```bash
python _smoke_test.py                                  # 합성 9클래스 생성
python extract_features.py --session smoke --out ft.csv
python run_ml.py --features ft.csv --out-prefix smoke
# 기대: 후각 ablation(−G)에서 Full > T+A 가 paired t-test 유의(p<0.05)
```

## 피처 구성 (42)
- 촉각 8, 음향 10, ENS160 8, BME688 12(단일스텝), 크로스모달 4 = **42**
- tap-only에서도 촉각 8개 전부 산출됨(steady 계열은 tap 윈도우 기준으로 계산,
  peak_cv는 3회 tap 반복성으로 오히려 안정적). rub 음향은 없으므로 tap 버스트로 추출.
- ablation은 모달 블록(T/A/G) 단위로 동작 → 피처 세부 개수 변화와 무관.

## 알아둘 점
- `run_experiment.py`의 변형 phase 이름(`tap`/`press`/`rub`)에 맞춰 구간을 자릅니다.
  변형을 바꾸면 해당 phase가 없을 때 관련 피처는 NaN(중앙값 대치)으로 처리됩니다.
- BME688 샘플 수는 펌웨어의 `GAS_PERIOD_MS`에 따라 달라지지만, 추출기는 샘플 수에
  무관하게 동작(있는 점으로 곡선 통계 계산)합니다.
- SHAP는 설치 시 자동 활성화: `pip install shap`.
