import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import mlflow.pytorch
from pathlib import Path
from datetime import datetime
import sys
from sklearn.preprocessing import MinMaxScaler

# 프로젝트 루트 경로 추가 (모듈 import용)
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from src.features.schema import FeatureSchema
from src.models.model_zoo import DeepCNN, CNNAttention, TransformerModel, DLinear, Simple1DCNN
from src.models.model_config import TRAINER_CONFIG, MODEL_CONFIGS # <--- Config 가져오기

# ==========================================
# 1. Config & Hyperparameters
# ==========================================
def get_model(model_name, input_dim, model_conf):
    """모델 이름과 설정값을 받아서 객체를 생성해주는 Factory 함수"""
    if model_name == "DLinear":
        return DLinear(seq_len=model_conf['window_size'], input_dim=input_dim)
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=model_conf['d_model'], nhead=model_conf['nhead'])
    elif model_name == "DeepCNN":
        return DeepCNN(input_dim=input_dim, hidden_dim=model_conf['hidden_dim'])
    
# ==========================================
# 3. Data Preparation (Sliding Window)
# ==========================================
def create_dataset(df, window_size, feature_cols):
    X_list, y_list = [], []
    
    # 엔진별로 윈도우 자르기
    for unit_nr, group in df.groupby('unit_nr'):
        data = group[feature_cols].values
        target = group['RUL'].values
        
        # 데이터가 윈도우보다 짧으면 패스
        if len(data) < window_size: continue
            
        # Sliding Window (속도 최적화 버전 아님, 이해용)
        for i in range(len(data) - window_size):
            X_list.append(data[i : i + window_size])
            y_list.append(target[i + window_size - 1]) # 윈도우 끝지점의 RUL 예측
            
    return np.array(X_list), np.array(y_list)

# ==========================================
# 4. Main Training Pipeline
# ==========================================
def train_model(model_name):
    # 1. 설정 로드 (공통 설정 + 모델 전용 설정 합체)
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 딕셔너리 병합 (Python 3.9+)
    config = TRAINER_CONFIG | MODEL_CONFIGS[model_name]
    
    # MLflow 세팅
    current_time = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{model_name}_{current_time}"
    mlflow.set_experiment("Turbofan_RUL_Prediction")

    with mlflow.start_run(run_name=run_name):
        print(f"🚀 Start Training: {model_name}")
        print(f"📜 Applied Config: {config}")
        mlflow.log_params(config) # 합쳐진 설정 기록

        # ----------------------------------------
        # 데이터 로드 & 전처리 (이전과 동일하지만 config 사용)
        # ----------------------------------------
        data_path = PROJECT_DIR / "data/processed/train_FD001_features.parquet"
        df = pd.read_parquet(data_path)
        
        # Scaling
        scaler = MinMaxScaler()
        df[config['features']] = scaler.fit_transform(df[config['features']])
        
        # Windowing (config['window_size'] 사용)
        # (create_dataset 함수는 기존과 동일하다고 가정)
        X, y = create_dataset(df, config['window_size'], config['features'])
        
        # DataLoader 생성
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        # ----------------------------------------
        # 모델 초기화 (Factory 함수 사용)
        # ----------------------------------------
        model = get_model(model_name, len(config['features']), MODEL_CONFIGS[model_name])
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # ----------------------------------------
        # 학습 루프
        # ----------------------------------------
        model.train()
        for epoch in range(config['epochs']):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            rmse = np.sqrt(avg_loss)
            
            # Scheduler Step
            scheduler.step(avg_loss)
            
            # Logging
            if (epoch+1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{config['epochs']} | RMSE: {rmse:.4f} | LR: {lr:.6f}")
                mlflow.log_metric("rmse", rmse, step=epoch)

        # 저장
        mlflow.pytorch.log_model(model, "model")
        print("🎉 Training Finished.")


if __name__ == "__main__":
    train_model(model_type="DLinear")