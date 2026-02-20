import json
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
PROJECT_DIR = BASE_DIR.parent

# 모델 클래스를 불러오기 위해 경로 추가 (PyTorch 모델 로드 시 필수)
import sys
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

# 원본 모델 아키텍처 임포트 (가중치를 덮어씌울 껍데기)
from src.models.model_zoo import TransformerModel 
from src.models.model_config import MODEL_CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전역 변수 (메모리 상주용)
preprocessors = {}
ensemble_models = []
max_window_size = 0

def load_artifacts():
    """서버 시작 시 메모리에 무거운 파일들을 1번만 로드합니다."""
    global preprocessors, ensemble_models, max_window_size
    
    print("⏳ [MLOps] Loading artifacts into memory...")
    
    # 1. 전처리기 로드
    preprocessors['pca_scaler'] = joblib.load(ARTIFACT_DIR / "pca_scaler.pkl")
    preprocessors['pca_model'] = joblib.load(ARTIFACT_DIR / "pca_model.pkl")
    preprocessors['minmax_scaler'] = joblib.load(ARTIFACT_DIR / "minmax_scaler.pkl")
    
    # 2. 메타데이터 로드
    with open(ARTIFACT_DIR / "ensemble_meta.json", "r") as f:
        meta = json.load(f)
        
    # 3. 모델 가중치(state_dict) 로드
    for model_info in meta["models"]:
        w_size = model_info["window_size"]
        max_window_size = max(max_window_size, w_size)
        
        # 모델 껍데기 생성 (학습 때와 동일한 하이퍼파라미터 필요 - 예시로 CONFIG 활용)
        # 만약 모델마다 파라미터가 달랐다면 meta.json에 파라미터도 같이 저장했어야 함
        model = TransformerModel(
            window_size=w_size, 
            d_model=MODEL_CONFIGS['Transformer']['d_model'],    # 추가됨!
            num_layers=MODEL_CONFIGS['Transformer']['num_layers'], # 변경됨!
            nhead=MODEL_CONFIGS['Transformer']['nhead'],           # 변경됨!
            dropout=0.0 # Option B: 추론 시에는 어차피 Dropout을 끕니다
        ).to(device)
        
        # 가중치 덮어쓰기
        weight_path = ARTIFACT_DIR / model_info["filename"]
        model.load_state_dict(torch.load(weight_path, map_location=device))
        
        # Option B: MC Dropout 끄고 일반 평가 모드 전환 (속도 극대화)
        model.eval() 
        
        ensemble_models.append({"model": model, "window_size": w_size})
        print(f"  ✅ Loaded {model_info['model_type']} (Window: {w_size})")

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """Raw 데이터를 모델 입력용으로 변환"""
    raw_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15']
    
    # 1. PCA 적용
    scaled_for_pca = preprocessors['pca_scaler'].transform(df[raw_sensors])
    df['pca_1'] = preprocessors['pca_model'].transform(scaled_for_pca)
    
    # 추론 시에는 이전 스텝과의 차이로 Trend를 구함 (간단한 diff 연산)
    df['pca_1_trend'] = df['pca_1'].diff().fillna(0)
    
    # 2. MinMax Scaling
    final_features = raw_sensors + ['pca_1', 'pca_1_trend']
    df[final_features] = preprocessors['minmax_scaler'].transform(df[final_features])
    
    return df[final_features].values

def predict_rul(raw_data_list: list) -> float:
    """Option B: 3개 모델 일반 추론 후 평균"""
    df = pd.DataFrame([vars(item) for item in raw_data_list])
    processed_data = preprocess_data(df) # (N_samples, 9)
    
    predictions = []
    
    with torch.no_grad(): # 역전파 계산 끔 (메모리 절약 & 속도 향상)
        for entry in ensemble_models:
            model = entry["model"]
            w_size = entry["window_size"]
            
            # 해당 모델의 윈도우 사이즈만큼 데이터 끝에서 잘라냄
            window_data = processed_data[-w_size:]
            
            # (1, Window_size, 9) 형태로 Tensor 변환
            X_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 예측 (1번만)
            pred = model(X_tensor).cpu().numpy().flatten()[0]
            predictions.append(pred)
            
    # 최종 평균 반환
    return float(np.mean(predictions))