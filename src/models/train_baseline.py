import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import mlflow.pytorch
import mlflow.data 
from mlflow.data.pandas_dataset import PandasDataset 
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view

PROJECT_DIR = Path(__file__).resolve().parents[2]

from src.features.schema import FeatureSchema
from src.models.model_zoo import DeepCNN, CNNAttention, TransformerModel, DLinear, Simple1DCNN
from src.models.model_config import TRAINER_CONFIG, MODEL_CONFIGS

def get_model(model_name, input_dim, model_conf):
    """모델 Factory 함수 (기존 동일)"""
    if model_name == "DLinear":
        return DLinear(seq_len=model_conf['window_size'], input_dim=input_dim)
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=model_conf['d_model'], nhead=model_conf['nhead'])
    elif model_name == "DeepCNN":
        return DeepCNN(input_dim=input_dim, hidden_layers=model_conf['hidden_layers'], kernel_size=model_conf['kernel_size'], dropout=model_conf['dropout'])
    elif model_name == "CNNAttention":
        return CNNAttention(input_dim=input_dim, hidden_dim=model_conf['hidden_dim'])
    else:
        return Simple1DCNN(input_dim=input_dim)

def create_dataset(df, window_size, feature_cols):
    """DataFrame -> Windowed Numpy Array (기존 동일)"""
    X_list, y_list = [], []
    
    # print(f"   [Info] Creating windows (Size: {window_size})...")
    
    for unit_nr, group in df.groupby('unit_nr'):
        data = group[feature_cols].values
        target = group['RUL'].values
        
        if len(data) < window_size:
            continue
            
        # Sliding Window
        windows = sliding_window_view(data, window_shape=window_size, axis=0)
        # Shape 변환: (N, Window, Feat) -> 모델 입력에 맞게 조정 (N, Feat, Window)가 필요하다면 transpose 위치 주의
        # 현재 DeepCNN 등은 (N, Feat, Window)를 기대하거나 내부에서 처리함.
        # 여기서는 (N, Window, Feat)로 유지하고 모델 내부에서 transpose한다고 가정하거나,
        # 기존 코드대로 (0, 2, 1) Transpose를 유지합니다.
        windows = windows.transpose(0, 2, 1) # (N, Feat, Window) 형태
        
        target_windows = target[window_size-1:]
        
        X_list.append(windows)
        y_list.append(target_windows)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def train_model(model_name):
    # 1. Config 병합
    if model_name not in MODEL_CONFIGS:
        print(f"⚠️ Warning: No specific config for {model_name}. Using default.")
        model_specific_conf = {}
    else:
        model_specific_conf = MODEL_CONFIGS[model_name]

    full_config = TRAINER_CONFIG.copy()
    full_config.update(model_specific_conf)
    full_config['model_type'] = model_name
    
    # 2. MLflow 설정
    current_time = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{model_name}_{current_time}"
    mlflow.set_experiment("Turbofan_RUL_Prediction")

    with mlflow.start_run(run_name=run_name):
        print(f"🚀 Start Training: {model_name}")
        print(f"📜 Full Config: {full_config}")
        mlflow.log_params(full_config)

        # ----------------------------------------
        # 3. 데이터 로드 및 분할 (핵심 수정!)
        # ----------------------------------------
        data_path = PROJECT_DIR / "data/processed/train_FD001_advanced_features.parquet"
        df = pd.read_parquet(data_path)


        ## RUL Clipping (최대 125까지만 예측하도록 제한)
        MAX_RUL = 125
        print(f" [Preprocessing] Clipping RUL to max {MAX_RUL}...")
        df['RUL'] = df['RUL'].clip(upper=MAX_RUL)
        
        # [Split Logic] Unit ID 기준 분할 (8:2)
        unit_ids = df['unit_nr'].unique()
        split_idx = int(len(unit_ids) * 0.8)
        train_units = unit_ids[:split_idx]
        val_units = unit_ids[split_idx:]
        
        print(f"   [Split] Train Units: {len(train_units)} / Val Units: {len(val_units)}")
        
        train_df = df[df['unit_nr'].isin(train_units)].copy()
        val_df = df[df['unit_nr'].isin(val_units)].copy()
        
        # MLflow Dataset Log (Train 기준)
        dataset = mlflow.data.from_pandas(train_df, source=str(data_path), name="turbofan_train_split")
        mlflow.log_input(dataset, context="training")

        # ----------------------------------------
        # 4. Scaling (Leakage 방지)
        # ----------------------------------------
        scaler = MinMaxScaler()
        feature_cols = full_config['features']
        
        # Train으로 fit, Val은 transform만!
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        
        # ----------------------------------------
        # 5. Windowing & DataLoader
        # ----------------------------------------
        # Train Set
        X_train, y_train = create_dataset(train_df, full_config['window_size'], feature_cols)
        train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_tensor, batch_size=full_config['batch_size'], shuffle=True)
        
        # Val Set
        X_val, y_val = create_dataset(val_df, full_config['window_size'], feature_cols)
        val_tensor = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
        val_loader = DataLoader(val_tensor, batch_size=full_config['batch_size'], shuffle=False) # 섞지 않음

        print(f"   [Data] Train Windows: {len(X_train)} / Val Windows: {len(X_val)}")

        # ----------------------------------------
        # 6. 모델 및 학습 설정
        # ----------------------------------------
        model = get_model(model_name, len(feature_cols), full_config)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=full_config['learning_rate'])
        patience = full_config.get('patience', 10) 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

        # ----------------------------------------
        # 7. 학습 루프 (Validation 추가)
        # ----------------------------------------
        print("🔥 Training Loop Start...")
        for epoch in range(full_config['epochs']):
            # --- Training ---
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_rmse = np.sqrt(avg_train_loss)

            # --- Validation ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_rmse = np.sqrt(avg_val_loss)
            
            # Scheduler Step (Validation 점수 기준)
            scheduler.step(avg_val_loss)
            
            # Logging
            if (epoch+1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{full_config['epochs']} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | LR: {lr:.6f}")
                
                mlflow.log_metric("train_rmse", train_rmse, step=epoch)
                mlflow.log_metric("val_rmse", val_rmse, step=epoch) # Val 점수가 진짜 중요함!
                mlflow.log_metric("learning_rate", lr, step=epoch)

        # 모델 저장
        mlflow.pytorch.log_model(model, "model")
        print(f"🎉 Training Finished. Final Val RMSE: {val_rmse:.4f}")

if __name__ == "__main__":
    train_model("DeepCNN")