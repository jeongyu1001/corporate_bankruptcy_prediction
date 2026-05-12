# corporate_bankruptcy_prediction
Bankruptcy prediction using tree-based classifiers on imbalanced financial data

# Corporate Bankruptcy Prediction

> 재무 특성을 이용한 기업 부실 예측 — 불균형 이진 분류 프로젝트

---

## 프로젝트 개요

재무제표 기반의 96개 변수를 활용해 기업의 부실(파산) 여부를 예측하는 이진 분류 모델을 구축합니다.  
데이터는 극단적인 클래스 불균형(정상 97% / 부실 3%)을 가지며, 이를 극복하기 위한 전처리 전략과 모델 선택 과정을 중점적으로 다룹니다.

- **핵심 과제:** 불균형 데이터에서 부실 기업 탐지율(Recall) 극대화

---

## 프로젝트 제약사항

| 항목 | 내용 |
|------|------|
| **허용 알고리즘** | `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier` |
| **제외 알고리즘** | XGBoost, LightGBM, CatBoost, Neural Network 등 외부 라이브러리 |
| **전처리** | 자유 (결측치 처리, 이상치 제거, 샘플링 등) |
| **Threshold** | `0.5` 고정 |
| **Random Seed** | `random_state=42` 고정 |
| **평가 지표** | M-Score = (Accuracy + Recall + Precision) / 3 |
| **지표 형식** | 소수점 형태 필수 (`0.85` ✅ / `85%` ❌) |

---

## 파일 구조

```
corporate_bankruptcy_prediction_PJ/
│
├── bankruptcy_prediction.ipynb   # 전체 분석 노트북 (메인)
├── model.pkl                     # 최종 저장 모델 (GradientBoostingClassifier)
├── README.md                     # 프로젝트 설명 (현재 파일)
│
├── classification_train_public.csv        # 학습 데이터 (비공개 — 미포함)
├── classification_test_private.csv        # 테스트 데이터 (비공개 — 미포함)
└── classification_description_v2.md      # 변수 설명 (비공개 — 미포함)
```

> **데이터 안내**  
> 데이터 파일은 수업 제공 비공개 자료로 이 레포지토리에 포함되지 않습니다.  
> 동일 데이터 보유 시 `bankruptcy_prediction.ipynb` 상단의 경로 설정 셀만 수정하면 전체 파이프라인이 재현됩니다.

---

## 분석 파이프라인

```
데이터 로드
    └─▶ 0→NaN 치환 (0 비율 60% 이상 컬럼)
            └─▶ Imputer (median)
                    └─▶ Winsorization (1%~99%)
                            └─▶ StandardScaler
                                    └─▶ SMOTE (train only, strategy=0.3)
                                            └─▶ 4개 모델 1차 비교
                                                    └─▶ 하이퍼파라미터 튜닝
                                                            └─▶ Feature Importance Top-70 선택
                                                                    └─▶ SelectFromModel 2차 선택 (~35개)
                                                                            └─▶ 최종 모델 학습 & 평가
```

---

## 전처리 전략

### 1. 0→NaN 치환 + Median Imputer
재무 데이터에서 0이 60% 이상인 컬럼은 실제 값이 아닌 **미기재/결측**일 가능성이 높다고 판단했습니다.  
해당 컬럼의 0을 NaN으로 치환한 뒤 `SimpleImputer(strategy='median')`으로 채웠습니다.  
평균 대신 중앙값을 사용한 이유는 재무 데이터의 극단값에 대한 **이상치 강건성** 때문입니다.

### 2. Winsorization (1% ~ 99%)
재무비율 특성상 극단값이 전체 분포를 왜곡할 수 있습니다.  
데이터를 완전히 제거하는 대신 상·하위 1% 경계값으로 **압축(clip)**하는 절충안을 선택했습니다.  
샘플 수(약 5,000개)가 많지 않아 정보 손실을 최소화하기 위한 판단입니다.

### 3. StandardScaler
SMOTE는 거리 기반으로 합성 샘플을 생성하므로, **스케일 차이가 크면 특정 변수 방향으로 왜곡**됩니다.  
Winsorization 이후 스케일을 통일해 SMOTE가 균등하게 작동하도록 했습니다.

### 4. SMOTE (Train only)
원본 클래스 비율: 정상 97% / 부실 3% → SMOTE 후: 정상 77% / 부실 23%  
`sampling_strategy=0.3`으로 부실 기업 비율을 정상 대비 30% 수준으로 조정했습니다.  
Validation에는 적용하지 않아 실제 운영 환경의 분포를 그대로 유지했습니다.

---

## 모델 선택 과정

| 단계 | 내용 |
|------|------|
| **1차 비교** | DT / RF / ET / GB 4종을 동일 조건으로 학습, M-Score 기준 비교 |
| **튜닝 제외** | DT — 단일 트리로 일반화 성능 열위 |
| **하이퍼파라미터 튜닝** | ET·RF: `GridSearchCV` / GB: `RandomizedSearchCV(n_iter=10)` |
| **튜닝 기준** | `scoring="f1"` — 불균형 데이터에서 Recall·Precision 균형 반영 |
| **최종 선택** | `GradientBoostingClassifier` (튜닝 후 M-Score 최고) |

---

## 최종 성능 (Validation 기준)

| 지표 | 값 |
|------|-----|
| Accuracy | — |
| Recall | ~0.33 |
| Precision | ~0.79 |
| **M-Score** | — |

> 결과 수치는 실행 환경에 따라 다를 수 있습니다.  
> 전체 결과는 노트북의 Step 6 출력 셀을 참고하세요.

---

## 회고 및 한계

### 잘 된 점
- 전처리 순서(Imputer → Winsorize → Scale → SMOTE)를 데이터 누수 없이 논리적으로 구성
- 4개 모델을 동일 조건으로 공정하게 비교
- PR Curve 채택 — 불균형 데이터에서 ROC보다 소수 클래스 성능을 정확히 반영

### 실패한 전략과 원인 분석

**핵심 문제: GB 모델 + 변수 선택의 불일치**

GB(Gradient Boosting)는 약한 신호도 순차적으로 학습하는 Boosting 계열 모델입니다.  
변수를 제거할수록 오히려 학습할 신호가 줄어 **Recall이 하락**하는 역효과가 발생했습니다.

반면 RF·ET는 랜덤 서브셋 기반이므로 노이즈 변수 제거 시 오히려 성능이 향상될 수 있었습니다.  
즉, **"변수 선택을 통한 Recall 향상"이라는 전략 자체는 GB가 아닌 RF·ET에 적합한 전략**이었습니다.

**추가 발견된 코드 오류**

하이퍼파라미터 튜닝 과정에서 ET·RF의 결과가 리스트에 저장되지 않는 버그로 인해,  
의도치 않게 GB가 최종 모델로 선택되었습니다.  
오류 없이 실행했다면 1차 비교 결과 상위 모델인 ET가 선택되었을 것입니다.

### 개선 방향

| 방향 | 설명 |
|------|------|
| 모델 교체 | GB 대신 RF 또는 ET + 변수 선택 조합 |
| Threshold 조정 | 0.5 → 0.3~0.4로 낮춰 Recall 향상 (단, 과제 조건상 0.5 고정) |
| 튜닝 지표 변경 | `scoring="f1"` → `scoring="recall"`로 변경해 부실 탐지에 집중 |
| 변수 전략 전환 | 변수 제거 대신 변수 확장 또는 SelectFromModel 미적용 |

---

## 실행 환경

```
Python       : 3.10+
scikit-learn : 1.6.1
imbalanced-learn : 최신
numpy, pandas, matplotlib, seaborn, joblib
```

> 정확한 버전은 노트북 마지막 셀 출력값을 확인하세요.

### Google Colab에서 실행하기

```python
# 1. Google Drive 마운트
from google.colab import drive
drive.mount("/content/drive")

# 2. 노트북 상단 경로 설정 셀에서 DATA_DIR만 수정
DATA_DIR = "/content/drive/MyDrive/corporate_bankruptcy_prediction_PJ/"
```

---

## 참고

- 평가 지표 M-Score = (Accuracy + Recall + Precision) / 3
- `model.pkl`은 sklearn 1.6.1 환경에서 저장되었습니다. 다른 버전 사용 시 경고가 발생할 수 있습니다.
- pkl 파일은 신뢰할 수 없는 환경에서의 실행을 권장하지 않습니다.
