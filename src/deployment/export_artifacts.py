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
    """
    [MLOps Architecture] Dynamic Metadata Extraction
    - 오리지널 앙상블 코드와 100% 동일하게 'val_rmse' 기준으로 Top 3를 선발합니다.
    - 선발된 모델들의 과거 하이퍼파라미터를 동적으로 추출하여 JSON에 저장합니다.
    
    ■ 도입 배경 (Why we do this?):
        - 문제점: 과거 학습된 모델(예: 128차원, 2레이어)의 가중치(.pth)를 
                 현재 업데이트된 코드(예: 256차원, 4레이어)의 모델 껍데기에 덮어씌우려 할 때 
                 차원 불일치(Shape Mismatch) 에러가 발생하는 '설정 표류(Configuration Drift)' 현상 발생.
        - 안티 패턴: 배포 서버(FastAPI) 개발자가 에러 로그를 보고 하드코딩으로 숫자를 맞춰줌. 
                   -> 모델이 재학습될 때마다 서버 코드를 수정해야 하는 치명적 의존성 발생.

    ■ 해결 기능 (What this does?):
        1. MLflow에서 순수 가중치(.pth)만 다운로드하는 것이 아님.
        2. 해당 가중치가 학습될 당시에 사용되었던 하이퍼파라미터(d_model, nhead, num_layers 등)를 
           MLflow 파라미터 기록에서 동적으로 함께 추출.
        3. 이 정보들을 'ensemble_meta.json'이라는 설계도 파일에 묶어서 배포.
        4. 서빙 서버는 이 JSON을 읽고 "스스로 알맞은 껍데기를 동적으로 생성"하게 됨.
           -> 모델 구조가 아무리 바뀌어도 서빙 코드는 단 한 줄도 수정할 필요 없는 완전 자동화 달성.
    
    """
    print(f"🚀 [MLOps] Extracting Top {top_n} Models (by Val RMSE) from MLflow...")
    
    mlflow.set_experiment("Turbofan_RUL_Prediction")
    experiment = mlflow.get_experiment_by_name("Turbofan_RUL_Prediction")
    
    # [수정됨] 오리지널 앙상블 로직과 완벽하게 동일한 쿼리 (Val RMSE 오름차순)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"], 
        max_results=top_n
    )
    
    ensemble_meta = {
        "models": [],
        "ensemble_method": "average",
        "expected_features": 9
    }
    
    for index, run in runs.iterrows():
        run_id = run.run_id
        model_name = run['params.model_type']
        window_size = int(run['params.window_size'])
        val_rmse = run['metrics.val_rmse']
        
        # [안전장치] DataFrame에 해당 파라미터 컬럼이 없거나 NaN일 경우 대비 (에러 방지)
        d_model = int(run['params.d_model']) if 'params.d_model' in run and pd.notna(run['params.d_model']) else 128
        nhead = int(run['params.nhead']) if 'params.nhead' in run and pd.notna(run['params.nhead']) else 4
        num_layers = int(run['params.num_layers']) if 'params.num_layers' in run and pd.notna(run['params.num_layers']) else 2
        
        print(f"  -> Exporting Rank {index+1}: {model_name} (W={window_size}, Val RMSE: {val_rmse:.4f})")
        print(f"     [Metadata] d_model={d_model}, nhead={nhead}, layers={num_layers}")
        
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
        weight_filename = f"model_rank{index+1}_w{window_size}.pth"
        weight_path = ARTIFACT_DIR / weight_filename
        
        torch.save(model.state_dict(), weight_path)
        
        ensemble_meta["models"].append({
            "rank": index + 1,
            "filename": weight_filename,
            "model_type": model_name,
            "window_size": window_size,
            "val_rmse": val_rmse,
            "run_id": run_id,
            "hyperparams": {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers
            }
        })

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