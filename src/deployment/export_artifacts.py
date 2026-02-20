import sys
from pathlib import Path

# 1. 경로 먼저 설정
PROJECT_DIR = Path(__file__).resolve().parents[2]

# 2. [가장 중요] 파이썬이 'src' 폴더를 찾을 수 있도록 경로를 강제로 주입
sys.path.append(str(PROJECT_DIR))

import os
import ast
import json
import torch
import joblib
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# 경로 설정
PROJECT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_DIR / "app/artifacts"  # FastAPI 도커가 읽을 폴더
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 기존 앙상블 로직 그대로 적용
def export_top_models(top_n=3):
    print(f"🚀 [MLOps] Extracting Top {top_n} Models (by MC Dropout Test RMSE) from MLflow...")
    
    mlflow.set_experiment("Turbofan_RUL_Prediction")
    experiment = mlflow.get_experiment_by_name("Turbofan_RUL_Prediction")
    
    # [수정된 핵심 로직] 
    # Val RMSE가 아니라, 우리가 앙상블 기준으로 삼았던 'test_rmse_mc_dropout' 오름차순으로 가져옵니다.
    # filter_string을 넣어 해당 메트릭이 없는(평가 안 한) 모델은 제외합니다.
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.test_rmse_mc_dropout > 0", 
        order_by=["metrics.test_rmse_mc_dropout ASC"], 
        max_results=top_n
    )
    
    ensemble_meta = {
        "models": [],
        "ensemble_method": "average",
        "expected_features": 9
    }
    
    # 1. 모델 가중치 및 메타데이터 추출
    for index, run in runs.iterrows():
        run_id = run.run_id
        model_name = run['params.model_type']
        window_size = int(run['params.window_size'])
        test_rmse_mc = run['metrics.test_rmse_mc_dropout']
        
        print(f"  -> Exporting Rank {index+1}: {model_name} (W={window_size}, MC Dropout RMSE: {test_rmse_mc:.4f})")
        
        # 모델 로드 후 순수 가중치(state_dict)만 추출
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
        weight_filename = f"model_rank{index+1}_w{window_size}.pth"
        weight_path = ARTIFACT_DIR / weight_filename
        
        torch.save(model.state_dict(), weight_path)
        
        # 메타데이터 기록
        ensemble_meta["models"].append({
            "rank": index + 1,
            "filename": weight_filename,
            "model_type": model_name,
            "window_size": window_size,
            "test_rmse_mc_dropout": test_rmse_mc, # Val 대신 Test 점수로 메타데이터 교체
            "run_id": run_id
        })

    # 메타데이터 JSON 저장
    with open(ARTIFACT_DIR / "ensemble_meta.json", "w") as f:
        json.dump(ensemble_meta, f, indent=4)
        
    print(f"✅ Models and metadata exported to {ARTIFACT_DIR}")

def export_preprocessors():
    print(f"🚀 [MLOps] Regenerating and Exporting Preprocessors (Scaler, PCA)...")
    
    # 훈련 데이터 로드 (정확히 훈련 때 사용한 그 데이터)
    train_path = PROJECT_DIR / "data/processed/train_FD001_advanced_features.parquet"
    train_df = pd.read_parquet(train_path)
    
    # 정예 9개 피처 (평가 코드에서 고정했던 그 피처들)
    # pca_1, pca_1_trend 생성을 위한 원본 센서들
    raw_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15']
    
    # 1. PCA 객체 생성 및 저장 (Fit)
    pca_scaler = StandardScaler()
    pca = PCA(n_components=1)
    
    train_scaled_for_pca = pca_scaler.fit_transform(train_df[raw_sensors])
    train_df['pca_1'] = pca.fit_transform(train_scaled_for_pca)
    train_df['pca_1_trend'] = train_df.groupby('unit_nr')['pca_1'].transform(lambda x: x.diff().fillna(0))
    
    joblib.dump(pca_scaler, ARTIFACT_DIR / "pca_scaler.pkl")
    joblib.dump(pca, ARTIFACT_DIR / "pca_model.pkl")
    
    # 2. MinMaxScaler 객체 생성 및 저장 (Fit)
    final_features = raw_sensors + ['pca_1', 'pca_1_trend']
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(train_df[final_features])
    
    joblib.dump(minmax_scaler, ARTIFACT_DIR / "minmax_scaler.pkl")
    
    print(f"✅ Preprocessors exported to {ARTIFACT_DIR}")

if __name__ == "__main__":
    export_top_models(top_n=3)
    export_preprocessors()
    print("🎉 All artifacts successfully packed for deployment!")