import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
import copy 

# 프로젝트 루트 경로 설정
PROJECT_DIR = Path(__file__).resolve().parents[2]

from src.features.schema import FeatureSchema
from src.models.model_zoo import DeepCNN, CNNAttention, TransformerModel, DLinear, Simple1DCNN
from src.models.model_config import TRAINER_CONFIG, MODEL_CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using Device: {device}")

def get_model(model_name, input_dim, model_conf):
    """모델 Factory 함수"""
    if model_name == "DLinear":
        model = DLinear(seq_len=model_conf['window_size'], input_dim=input_dim)
    elif model_name == "Transformer":
        model = TransformerModel(input_dim=input_dim, d_model=model_conf['d_model'], nhead=model_conf['nhead'])
    elif model_name == "DeepCNN":
        model = DeepCNN(input_dim=input_dim, hidden_layers=model_conf['hidden_layers'], kernel_size=model_conf['kernel_size'], dropout=model_conf['dropout'])
    elif model_name == "CNNAttention":
        model = CNNAttention(input_dim=input_dim, hidden_dim=model_conf['hidden_dim'])
    else:
        model = Simple1DCNN(input_dim=input_dim)
    
    return model.to(device)

def create_dataset(df, window_size, feature_cols):
    X_list, y_list = [], []
    for unit_nr, group in df.groupby('unit_nr'):
        data = group[feature_cols].values
        target = group['RUL'].values
        if len(data) < window_size: continue
        windows = sliding_window_view(data, window_shape=window_size, axis=0)
        windows = windows.transpose(0, 2, 1) 
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
    # full_config['epochs'] = 70 # 100은 너무 기니까 60으로 압축해서 스케줄링
    
    # 2. MLflow 설정
    current_time = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{model_name}_{current_time}_Modern"
    mlflow.set_experiment("Turbofan_RUL_Prediction")

    with mlflow.start_run(run_name=run_name):
        print(f"🚀 Start Training: {model_name}")
        print(f"📜 Full Config: {full_config}")
        mlflow.log_params(full_config)

        # 3. 데이터 로드
        data_path = PROJECT_DIR / "data/processed/train_FD001_advanced_features.parquet"
        df = pd.read_parquet(data_path)
        
        dataset = mlflow.data.from_pandas(df, source=str(data_path), name="turbofan_train_split")
        mlflow.log_input(dataset, context="training")

        MAX_RUL = 125
        print(f" [Preprocessing] Clipping RUL to max {MAX_RUL}...")
        df['RUL'] = df['RUL'].clip(upper=MAX_RUL)
        
        unit_ids = df['unit_nr'].unique()
        split_idx = int(len(unit_ids) * 0.8)
        train_units = unit_ids[:split_idx]
        val_units = unit_ids[split_idx:]
        
        train_df = df[df['unit_nr'].isin(train_units)].copy()
        val_df = df[df['unit_nr'].isin(val_units)].copy()
        
        scaler = MinMaxScaler()
        feature_cols = full_config['features']
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        
        X_train, y_train = create_dataset(train_df, full_config['window_size'], feature_cols)
        X_val, y_val = create_dataset(val_df, full_config['window_size'], feature_cols)
        
        train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_tensor = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
        
        train_loader = DataLoader(train_tensor, batch_size=full_config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_tensor, batch_size=full_config['batch_size'], shuffle=False, num_workers=0)

        print(f"   [Data] Train Windows: {len(X_train)} / Val Windows: {len(X_val)}")

        # 6. 모델 및 학습 설정
        model = get_model(model_name, len(feature_cols), full_config)
        criterion = nn.MSELoss()
        
        # [수정 1] Optimizer: AdamW + 초기 LR 안전하게 낮춤 (1e-3)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        # [수정 2] Scheduler: OneCycleLR (Warmup 내장 + 배치 단위 업데이트)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,              # 최대 학습률 (0.001 추천)
            epochs=full_config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,            # 10% 기간 동안 웜업
            anneal_strategy='cos'
        )
        
        patience_stop = 15 # 좀 더 오래 돌게 셋팅
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())

        # 7. 학습 루프
        print("🔥 Training Loop Start (with OneCycleLR & GradClip)...")
        
        for epoch in range(full_config['epochs']):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # [추가] Gradient Clipping: 기울기가 너무 커지지 않게 자름 (폭발 방지)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                # [이동] Scheduler는 배치마다 실행!
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_rmse = np.sqrt(avg_train_loss)

            # --- Validation ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_rmse = np.sqrt(avg_val_loss)
            
            # [삭제] 기존 scheduler.step(avg_val_loss)는 삭제됨 (OneCycleLR은 자동임)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{full_config['epochs']} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | LR: {current_lr:.6f}")
            
            mlflow.log_metric("train_rmse", train_rmse, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # --- Early Stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                print(f"  --> ⭐ New Best Model! (Val RMSE: {val_rmse:.4f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience_stop:
                    print(f"🛑 Early Stopping Triggered at Epoch {epoch+1}")
                    break

        print(f"💾 Restoring Best Model Weights (Val RMSE: {np.sqrt(best_val_loss):.4f})...")
        model.load_state_dict(best_model_wts)
        mlflow.pytorch.log_model(model, "model")
        print(f"🎉 Training Finished Successfully.")

if __name__ == "__main__":
    train_model("Transformer")