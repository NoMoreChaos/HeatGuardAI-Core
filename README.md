# HeatGuardAI

서울시 폭염 대응을 위한 AI 기반 쿨링포그 최적 설치 위치 추천 시스템

## 개요

HeatGuardAI는 서울 전역의 100m×100m 격자 단위로 쿨링포그 설치 우선순위를 산출하는 도시 열환경 분석 시스템입니다. 위성 데이터, 기상 데이터, 유동인구, 취약계층 정보를 통합하고 LLM 기반 전문가 설문을 통해 도출한 가중치를 적용하여 근거 기반의 설치 위치를 추천합니다.

## 시스템 구조

```
HeatGuardAI-Core/
    ├── 01_build_grid_100m_seoul.ipynb       # 서울 100m 격자 생성
    ├── 02_map_lst_to_grid.ipynb             # 지표면 온도(LST) 매핑
    ├── 03_map_gu_to_grid.ipynb              # 자치구 매핑
    ├── 04_map_dong_to_grid.ipynb            # 행정동 매핑
    ├── 05_lst_outlier_check.ipynb           # LST 이상치 검사
    ├── 06_compute_suhi_from_lst.ipynb       # 도시열섬(SUHI) 지수 산출
    ├── 07_map_ndvi_to_grid.ipynb            # 식생지수(NDVI) 매핑
    ├── 08_map_vulnerable_to_grid.ipynb      # 취약계층 분포 매핑
    ├── 09_get_ta_chi.ipynb                  # 기상청 체감온도 수집
    ├── 10_map_ta_chi_to_grid.ipynb          # 체감온도 보간 매핑
    ├── 11_make_buspop_data.ipynb            # 버스 승하차 → 유동인구 가공
    ├── 12_map_buspop_to_grid.ipynb          # 유동인구 격자 매핑
    ├── 13_merge_cellid_admin_lst_suhii.ipynb# 피처 통합
    ├── 14_피처엔지니어링_점수화.ipynb       # 피처 정규화 (0~1 점수)
    ├── 15_한강제거.ipynb                    # 탐색적 분석 (GMM 클러스터링)
    ├── 16_fog_scored_refactored.ipynb       # 가중 합산 및 최종 순위 산출
    └── llm/
        ├── src/
        │   ├── run_survey.py                # 전문가 설문 실행
        │   ├── aggregate.py                 # 설문 가중치 집계
        │   ├── add_llm_summary.py           # 추천 설명문 생성
        │   └── llm_summary_postprocess.py   # 설명문 후처리
        └── survey/
            ├── survey.json                  # 설문 구조 정의
            └── personas.json               # 전문가 페르소나 정의
```

## 데이터 흐름

```
원천 데이터
├── Landsat 위성 (LST, NDVI)
├── 기상청 API (체감온도)
├── 서울시 버스 승하차 데이터 (유동인구)
├── 서울시 취약계층 현황 (노인, 아동)
└── 행정 경계 Shapefile
        ↓
[노트북 01-13] 격자 생성 및 피처 매핑
        ↓
heatguard_cell_feature_scored_clean.csv
        ↓
[노트북 14] 피처 정규화 → 0~1 점수화
        ↓
[LLM 설문] 전문가 가중치 도출 (OpenAI API)
        ↓
[노트북 16] 가중 합산 → 우선순위 순위 산출
        ↓
heatguard_ranking_data_cluster_1.csv (최종 결과)
```

## 사용 피처 및 가중치

| 피처 | 설명 | 가중치 |
|------|------|--------|
| 취약계층 (취약인구) | 노인·아동 등 취약계층 밀집도 | 24.6% |
| SUHII (도시열섬지수) | Landsat 기반 지표면 도시열섬 강도 | 22.7% |
| 체감온도 | 기상청 체감온도 (IDW 보간) | 21.5% |
| NDVI (식생지수, 역산) | 녹지 부족 정도 (낮을수록 우선순위 높음) | 17.0% |
| 유동인구 | 버스 승하차 기반 보행자 밀도 | 14.2% |

가중치는 LLM 기반 전문가 설문(`llm/` 모듈)을 통해 6개 전문가 페르소나의 응답을 집계하여 산출합니다.

## 점수화 방법론

1. **피처 정규화**: 각 원시 피처를 MinMax 또는 IQR 기반으로 0~1 스케일로 변환
2. **Robust Z-score**: 최종 가중 합산 시 중앙값/MAD 기반 z-score 사용 (이상치 영향 최소화)
3. **클리핑**: z-score를 [-3, +3] 범위로 제한
4. **공간 필터링**: 동일 행정동 내 200m 이내 중복 추천 제거

## 전문가 설문 시스템 (LLM)

6개 전문가 페르소나가 8개 섹션·10개 문항으로 구성된 설문에 응답하며, 각 피처의 중요도 가중치를 도출합니다.

**전문가 페르소나:**
- 도시기후 연구자 — 열 지수 및 과학적 재현성 중시
- 정책 담당자 — 시민 체감 및 실행 가능성 중시
- 공중보건 전문가 — 취약계층 보호 중시
- 환경·생태 전문가 — 자연 냉각 효과(녹지) 중시
- 인프라 엔지니어 — 설치 적합성 및 유지보수 중시
- 공간·데이터 분석가 — 100m 해상도 적합성 및 중복 계산 방지 중시

### 설문 실행

```bash
cd HeatGuardAI-Core/llm

# 1. 전문가 설문 실행 (6개 페르소나)
python src/run_survey.py

# 2. 가중치 집계
python src/aggregate.py

# 3. (선택) 추천 설명문 생성
python src/add_llm_summary.py
python src/llm_summary_postprocess.py
```

**환경 변수:**

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4.1"   # 기본값: gpt-4.1-mini
```

## 설치 및 실행

### 의존성 설치

```bash
pip install pandas numpy scikit-learn matplotlib jupyter openai
```

### 전체 파이프라인 실행

```bash
cd HeatGuardAI-Core

# Jupyter 노트북 실행 (01 → 16 순서로 실행)
jupyter notebook
```

노트북을 `01_build_grid_100m_seoul.ipynb`부터 `16_fog_scored_refactored.ipynb`까지 순서대로 실행합니다.

### 최종 결과 확인

`16_fog_scored_refactored.ipynb` 실행 후 생성되는 `heatguard_ranking_data_cluster_1.csv`에서 격자 셀별 우선순위 점수와 행정 위치 정보를 확인할 수 있습니다.

## 출력 결과

| 컬럼 | 설명 |
|------|------|
| `GlobalScore` | 가중 합산 종합 우선순위 점수 |
| `zscore_suhi` | SUHI z-score |
| `zscore_ta` | 체감온도 z-score |
| `zscore_pop` | 유동인구 z-score |
| `zscore_ndvi` | NDVI(역산) z-score |
| `zscore_vul` | 취약계층 z-score |
| `gu_name` | 자치구 |
| `dong_name` | 행정동 |
| `cell_id` | 100m 격자 셀 ID |

## 기술 스택

- **Python 3** — pandas, numpy, scikit-learn, matplotlib
- **Jupyter Notebook** — 단계별 데이터 처리 파이프라인
- **OpenAI API** — 전문가 설문 자동화 및 설명문 생성
- **Landsat 위성 데이터** — LST, NDVI
- **기상청 Open API** — 체감온도
- **서울 공공데이터** — 버스 승하차, 취약계층, 행정 경계

## 관련 링크

- GitHub: [NoMoreChaos/HeatGuardAI-Core](https://github.com/NoMoreChaos/HeatGuardAI-Core)
