import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import mlflow.pytorch
import mlflow.data # <--- Dataset 로깅용 import
from mlflow.data.pandas_dataset import PandasDataset # <--- 명시적 import
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view


PROJECT_DIR = Path(__file__).resolve().parents[2]

# 분리한 모듈 임포트
from src.features.schema import FeatureSchema
from src.models.model_zoo import DeepCNN, CNNAttention, TransformerModel, DLinear, Simple1DCNN
from src.models.model_config import TRAINER_CONFIG, MODEL_CONFIGS




def get_model(model_name, input_dim, model_conf):
    """모델 이름과 설정값을 받아서 객체를 생성해주는 Factory 함수"""
    
    if model_name == "DLinear":
        return DLinear(seq_len=model_conf['window_size'], input_dim=input_dim)
    
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=model_conf['d_model'], nhead=model_conf['nhead'])
    
    elif model_name == "DeepCNN":
        return DeepCNN(input_dim=input_dim, hidden_layers=model_conf['hidden_layers'], kernel_size=model_conf['kernel_size'], dropout=model_conf['dropout'])
    
    elif model_name == "CNNAttention": # CNNAttention도 추가
        return CNNAttention(input_dim=input_dim, hidden_dim=model_conf['hidden_dim'])
    
    else:
        return Simple1DCNN(input_dim=input_dim)



def train_model(model_name):
    # ---------------------------------------------------------
    # 1. Config 합치기 (Merge Logic)
    # ---------------------------------------------------------
    if model_name not in MODEL_CONFIGS:
        # 모델별 설정이 없으면 기본 빈 딕셔너리라도 사용하거나 에러 처리
        # 여기서는 TRAINER_CONFIG만 사용하도록 처리할 수도 있음
        print(f"⚠️ Warning: No specific config for {model_name}. Using default.")
        model_specific_conf = {}
    else:
        model_specific_conf = MODEL_CONFIGS[model_name]

    # [핵심] 두 딕셔너리 병합 (.copy()로 원본 오염 방지)
    # full_config = 공통 설정 + 모델별 설정 + 모델 이름
    full_config = TRAINER_CONFIG.copy()
    full_config.update(model_specific_conf)
    full_config['model_type'] = model_name # 이름도 명시적으로 기록
    
    # ---------------------------------------------------------
    # MLflow 세팅
    # ---------------------------------------------------------
    current_time = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{model_name}_{current_time}"
    mlflow.set_experiment("Turbofan_RUL_Prediction")

    with mlflow.start_run(run_name=run_name):
        print(f"🚀 Start Training: {model_name}")
        print(f"📜 Full Config: {full_config}")
        
        # [핵심] 합쳐진 설정을 기록 -> 이제 Window Size 보입니다!
        mlflow.log_params(full_config) 

        # ----------------------------------------
        # 데이터 로드
        # ----------------------------------------
        data_path = PROJECT_DIR / "data/processed/train_FD001_features.parquet"
        df = pd.read_parquet(data_path)
        
        # ---------------------------------------------------------
        # [핵심] Dataset 정보 MLflow에 등록 (Data Lineage)
        # ---------------------------------------------------------
        print("[Info] Logging dataset info to MLflow...")
        dataset = mlflow.data.from_pandas(
            df, 
            source=str(data_path), 
            name="turbofan_processed_data_ver_1"
        )
        mlflow.log_input(dataset, context="training")
        # ---------------------------------------------------------
        
        # Scaling
        scaler = MinMaxScaler()
        # 주의: config에 있는 features 리스트만 사용
        feature_cols = full_config['features']
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Windowing (merged config에서 window_size 가져옴)
        X, y = create_dataset(df, full_config['window_size'], feature_cols)
        
        # DataLoader 생성
        dataset_tensor = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
        dataloader = DataLoader(dataset_tensor, batch_size=full_config['batch_size'], shuffle=True)

        # ----------------------------------------
        # 모델 초기화
        # ----------------------------------------
        model = get_model(model_name, len(feature_cols), full_config)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=full_config['learning_rate'])
        
        # Patience도 config에서 가져오기
        patience = full_config.get('patience', 10) 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

        # ----------------------------------------
        # 학습 루프
        # ----------------------------------------
        model.train()
        for epoch in range(full_config['epochs']):
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
                print(f"Epoch {epoch+1}/{full_config['epochs']} | RMSE: {rmse:.4f} | LR: {lr:.6f}")
                mlflow.log_metric("rmse", rmse, step=epoch)
                mlflow.log_metric("learning_rate", lr, step=epoch)

        # 저장
        mlflow.pytorch.log_model(model, "model")
        print("🎉 Training Finished.")

def create_dataset(df, window_size, feature_cols):
    X_list, y_list = [], []
    
    print(f"[Info] Creating windows (Size: {window_size})...") # 진행상황 출력
    
    for unit_nr, group in df.groupby('unit_nr'):
        data = group[feature_cols].values
        target = group['RUL'].values
        
        if len(data) < window_size:
            continue
            
        # 🚀 [NumPy Magic] 반복문 없이 한방에 자르기
        # sliding_window_view: 메모리 복사 없이 뷰만 생성해서 엄청 빠름
        # shape: (num_windows, window_size, num_features)
        windows = sliding_window_view(data, window_shape=window_size, axis=0).transpose(0, 2, 1)
        
        # y값은 각 윈도우의 '마지막 시점'의 RUL
        # target[window_size-1 :] 과 동일
        target_windows = target[window_size-1:]
        
        X_list.append(windows)
        y_list.append(target_windows)
        
    # 리스트 합치기
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"[Info] Windowing Complete! Shape: {X.shape}") # 완료 메시지
    return X, y

if __name__ == "__main__":
    # 원하는 모델로 테스트
    train_model("DeepCNN")