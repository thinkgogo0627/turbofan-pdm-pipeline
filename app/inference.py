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
    """
    [MLOps Architecture] Dynamic Model Factory (서버 시작 시 1회 동작)

    ■ 기능 설명 (What this does?):
        - 이 함수는 하드코딩된 파라미터(예: d_model=128)를 절대 사용하지 않습니다.
        - export_artifacts.py가 포장해준 'ensemble_meta.json' 설계도를 읽어들입니다.
        - JSON에 적힌 hyperparams를 바탕으로, 각 모델이 과거에 학습되었던 
          정확한 규격(Shape)의 껍데기를 메모리에 동적으로 찍어냅니다.
        
    ■ 기대 효과 (Impact):
        - 향후 데이터 사이언티스트가 트랜스포머 레이어를 100층으로 늘리든, 
          차원을 1024로 늘리든 서빙(FastAPI) 엔지니어는 코드를 건드릴 필요가 없습니다.
        - 지속적 배포(CD, Continuous Deployment) 파이프라인의 핵심 기반이 됩니다.
    """
    global preprocessors, ensemble_models, max_window_size
    
    print("⏳ [MLOps] Reading dynamic metadata & Loading artifacts into memory...")
    
    preprocessors['pca_scaler'] = joblib.load(ARTIFACT_DIR / "pca_scaler.pkl")
    preprocessors['pca_model'] = joblib.load(ARTIFACT_DIR / "pca_model.pkl")
    preprocessors['minmax_scaler'] = joblib.load(ARTIFACT_DIR / "minmax_scaler.pkl")
    
    with open(ARTIFACT_DIR / "ensemble_meta.json", "r") as f:
        meta = json.load(f)
        
    for model_info in meta["models"]:
        w_size = model_info["window_size"]
        max_window_size = max(max_window_size, w_size)
        
        weight_path = ARTIFACT_DIR / model_info["filename"]
        
        # 1. 껍데기를 만들기 '전'에 가중치 파일(.pth)을 먼저 뜯어봄 (현물 확인)
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        
        # 2. [역공학] 가중치 텐서의 형태(Shape)에서 진짜 아키텍처 규격을 알아냄
        # - embedding.weight의 크기는 [d_model, input_dim] 임. 여기서 d_model 훔쳐오기!
        d_model_inferred = state_dict['embedding.weight'].shape[0]
        
        # - transformer_encoder.layers.X 중에 가장 큰 층수(X)를 찾아서 +1 하기!
        layer_keys = [int(k.split('.')[2]) for k in state_dict.keys() if 'transformer_encoder.layers.' in k]
        num_layers_inferred = max(layer_keys) + 1 if layer_keys else 2
        
        # - nhead는 가중치 모양에서 직접 보이지 않으므로 JSON 값을 쓰되, 에러 방지용 안전장치 추가
        nhead_inferred = model_info.get("hyperparams", {}).get("nhead", 4)
        if d_model_inferred % nhead_inferred != 0: 
            nhead_inferred = 4 # nhead는 반드시 d_model의 약수여야 함

        print(f"  🔍 [Reverse Engineering] Inferred Spec -> d_model: {d_model_inferred}, layers: {num_layers_inferred}")
        
        # 3. 알아낸 '진짜' 규격으로 동적 껍데기 생성!
        model = TransformerModel(
            input_dim=9, 
            d_model=d_model_inferred,
            nhead=nhead_inferred,
            num_layers=num_layers_inferred
        ).to(device)
        
        # 4. 완벽하게 맞춰진 껍데기에 가중치 덮어쓰기
        model.load_state_dict(state_dict)
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