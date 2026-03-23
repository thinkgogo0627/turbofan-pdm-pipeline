# ⚙️ Turbofan PdM Pipeline

**NASA CMAPSS 터보팬 엔진 데이터셋 기반 예지보전(PdM) MLOps 파이프라인**

PyTorch로 학습한 RUL(Remaining Useful Life) 예측 모델을 FastAPI로 서빙하고, Streamlit 대시보드에서 SPC(Statistical Process Control) 실시간 모니터링까지 한 번에 경험할 수 있는 End-to-End MLOps 프로젝트입니다.

-----

## 🗂️ 프로젝트 구조

```
turbofan-pdm-pipeline/
├── src/
│   ├── data/           # 데이터 전처리 스크립트
│   ├── features/       # 피처 엔지니어링
│   ├── models/         # 모델 정의 및 학습 코드
│   └── evaluate/       # 모델 평가
├── app/
│   └── main.py         # FastAPI 서버 (POST /predict)
├── data/
│   └── processed/      # DVC로 추적되는 전처리 완료 데이터
├── .dvc/               # DVC 메타데이터
├── dashboard.py        # Streamlit 실시간 모니터링 대시보드
├── Dockerfile          # API 서버 컨테이너 이미지 빌드
├── config.yaml         # 모델 및 파이프라인 하이퍼파라미터
└── requirements.txt    # Python 의존성
```

-----

## 🔍 핵심 기능

|기능                |설명                                 |
|------------------|-----------------------------------|
|**RUL 예측 API**    |슬라이딩 윈도우(70 cycles) 기반 잔여 수명 예측    |
|**SPC 모니터링**      |UCL / LCL (±3σ) 기반 이탈 감지 및 실시간 경보  |
|**멀티 데이터셋**       |FD001 ~ FD004 4가지 운전 조건 동시 지원      |
|**API Latency 추적**|대시보드에서 추론 지연 시간 실시간 표시             |
|**Docker 배포**     |단일 `docker build` 명령으로 API 서버 컨테이너화|
|**DVC 데이터 관리**    |전처리 데이터 버전 추적 및 재현성 확보             |

-----

## 🧠 모델 & 데이터

### 데이터셋

[NASA CMAPSS Turbofan Engine Degradation Simulation Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

|서브셋  |운전 조건|고장 모드|
|-----|-----|-----|
|FD001|단일 조건|1가지  |
|FD002|다중 조건|1가지  |
|FD003|단일 조건|2가지  |
|FD004|다중 조건|2가지  |

### 사용 센서 피처

`sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_11`, `sensor_12`, `sensor_15`

### 예측 방식

- 입력: 최근 **70 cycles** 슬라이딩 윈도우
- 출력: 해당 시점의 **잔여 수명(RUL)** 예측값
- SPC: 정상 상태 기준 `mean=60`, `σ=4` → UCL=72, LCL=48

-----

## 🚀 Quick Start

### 1. 사전 요구사항

- Python 3.13+
- Docker (API 서버 실행 시)

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터 준비

```bash
# NASA CMAPSS 데이터를 data/raw/ 에 배치 후
dvc repro   # 전처리 파이프라인 실행
```

### 4. 모델 학습

```bash
PYTHONPATH=. python src/models/train.py
```

### 5. API 서버 실행

**로컬 실행:**

```bash
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Docker 실행:**

```bash
docker build -t turbofan-pdm .
docker run -p 8000:8000 turbofan-pdm
```

### 6. 대시보드 실행

> API 서버가 `localhost:8000`에서 실행 중이어야 합니다.

```bash
streamlit run dashboard.py
```

-----

## 🌐 API 명세

### `POST /predict`

엔진 센서 데이터를 받아 RUL을 예측합니다.

**Request Body**

```json
{
  "unit_id": 1,
  "cycle": 150,
  "data": [
    {
      "time_cycles": 81,
      "sensor_2": 641.82,
      "sensor_3": 1589.70,
      "sensor_4": 1400.60,
      "sensor_7": 554.36,
      "sensor_11": 47.47,
      "sensor_12": 521.66,
      "sensor_15": 8.4195
    }
    // ... 70 cycles worth of data
  ]
}
```

**Response**

```json
{
  "unit_id": 1,
  "cycle": 150,
  "predicted_rul": 42.3
}
```

-----

## 📊 대시보드 미리보기

대시보드는 FD001 ~ FD004 각각을 탭으로 구성하며, Engine ID를 선택해 사이클별 RUL 추이와 SPC 경보를 실시간으로 확인할 수 있습니다.

```
⚙️ Turbofan Engine RUL & SPC Monitoring System
├── [FD001] [FD002] [FD003] [FD004]  ← 데이터셋 탭
│
├── Engine ID 선택
├── 실시간 메트릭 (Cycle / Predicted RUL / SPC Status / API Latency)
├── RUL 추이 라인차트 (UCL · LCL 포함)
└── 🚨 SPC 이탈 시 실시간 경보
```

-----

## 🛠️ 기술 스택

|분류          |기술                                      |
|------------|----------------------------------------|
|**모델링**     |PyTorch 2.6, scikit-learn, pandas, numpy|
|**서빙**      |FastAPI, Uvicorn                        |
|**대시보드**    |Streamlit                               |
|**컨테이너**    |Docker (python:3.13-slim)               |
|**데이터 버전관리**|DVC                                     |
|**유틸리티**    |joblib                                  |

-----

## 📁 데이터 디렉토리 구조 (로컬)

```
data/
├── raw/              # NASA 원본 데이터 (git-ignored)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ... (FD002~FD004 동일 구조)
└── processed/        # DVC 추적 전처리 데이터
```

> ⚠️ 원본 데이터는 저작권상 레포에 포함되지 않습니다. [NASA PCoE](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)에서 직접 다운로드하세요.

-----

## 📌 향후 개선 방향

- [ ] MLflow 실험 추적 연동
- [ ] GitHub Actions CI/CD 파이프라인 구성
- [ ] Prometheus + Grafana 모델 드리프트 모니터링
- [ ] 다중 모델 A/B 테스팅 엔드포인트 추가

-----

## 📄 License

MIT License