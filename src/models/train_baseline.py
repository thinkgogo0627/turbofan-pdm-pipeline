import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

from src.features.schema import FeatureSchema # 방금 만든 스키마
from src.models.model_zoo import DeepCNN, CNNAttention, TransformerModel, DLinear


# ==========================================
# 1. Config & Hyperparameters
# ==========================================
params = {
    "window_size": 30,    # 과거 30초를 봄
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 10,
    "features": [
        'sensor_2_ema', 'sensor_3_ema', 'sensor_4_ema', 'sensor_7_ema',
        'sensor_11_ema', 'sensor_12_ema', 'sensor_15_ema' 
        # (Variation 컬럼들도 추가 가능)
    ]
}

# ==========================================
# 2. Model Architecture (Simple 1D CNN)
# ==========================================
class Simple1DCNN(nn.Module):
    def __init__(self, input_dim):
        super(Simple1DCNN, self).__init__()
        # Conv1d: 시계열의 '지역적 패턴'을 찾음 (필터가 시간축으로 슬라이딩)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Flatten 후의 차원 계산이 귀찮으므로 AdaptiveAvgPool 사용 (꼼수)
        # 어떤 길이든 1개의 값으로 압축
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(32, 1) # RUL 예측 (Regression)

    def forward(self, x):
        # x shape: (Batch, Time, Features) -> (Batch, Features, Time) 변환 필요
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.global_pool(x) # (Batch, 32, 1)
        x = self.flatten(x)     # (Batch, 32)
        x = self.fc(x)          # (Batch, 1)
        return x

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
def train(model_type):

    # 이름 생성
    current_time = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{model_type}_{current_time}"

    # MLflow 실험 이름 설정
    mlflow.set_experiment("Turbofan_RUL_Prediction")

    
    with mlflow.start_run(run_name=run_name):
        # A. 데이터 로드 및 검증 (Pandera)
        print("[Step 1] Loading & Validating Data...")
        data_path = PROJECT_DIR / "data/processed/train_FD001_features.parquet"
        df = pd.read_parquet(data_path)

    
        ## MLflow에 데이터셋 정보 등록
        print("[Info] logging dataset info to mlflow")
        # pandas dataframe -> mlflow dataset 객체 변환
        dataset = mlflow.data.from_pandas(
            df,
            source=str(data_path),
            name = "turbofan_processed_data_ver_1"
        )
        # train 용도로 사용했다고 기록
        mlflow.log_input(dataset, context="training")
        
        # Pandera 검증 수행 (실패하면 에러 발생)
        try:
            FeatureSchema.validate(df)
            print("✅ Data Schema Validation Passed!")
        except Exception as e:
            print(f"❌ Data Validation Failed: {e}")
            return

        ### Pandera 데이터 무결성 검증 후 Scaling 수행
        print("[Step 1.5] Applying MinMaxScaler")

        # Feature 컬럼 , Target 컬럼 분리
        feature_cols = params['features']

        # 스케일러 정의
        scaler = MinMaxScaler()

        # 데이터프레임의 Feature 만 스케일링 -> Target은 스케일링 X
        df[feature_cols] = scaler.fit_transform(df[feature_cols])


        # B. 전처리 (Windowing)
        print("[Step 2] Creating Sliding Windows...")
        X, y = create_dataset(df, params['window_size'], params['features'])
        
        # Tensor 변환
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        
        # C. 모델 초기화
        if model_type == "DeepCNN":
            model = DeepCNN(input_dim=len(params['features']))
        
        elif model_type == "CNNAttention":
            model = CNNAttention(input_dim=len(params['features']))
        
        elif model_type == "Transformer":
            model = TransformerModel(input_dim=len(params['features']))

        elif model_type == "DLinear":
            model = DLinear(seq_len=params['window_size'], input_dim=len(params['features']))
        
        else:
            model = Simple1DCNN(input_dim=len(params['features'])) # Default

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # MLflow 파라미터 기록
        mlflow.log_params(params)
        
        # D. 학습 루프
        print("[Step 3] Training Start...")
        model.train()
        for epoch in range(params['epochs']):
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
            print(f"Epoch {epoch+1}/{params['epochs']}, Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}")
            
            # MLflow 메트릭 기록
            mlflow.log_metric("rmse", rmse, step=epoch)
            
        # E. 모델 저장
        print("[Step 4] Saving Model...")
        mlflow.pytorch.log_model(model, "model")
        print("🎉 Training Complete! Check MLflow UI.")

if __name__ == "__main__":
    train(model_type="DLinear")