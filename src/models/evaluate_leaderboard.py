import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import ast

# 프로젝트 경로 설정
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

# ==========================================
# 1. Feature Engineering Logic
# ==========================================
def apply_ema(df, sensors, alpha=0.1):
    """EMA 피처 생성"""
    for sensor in sensors:
        df[f"{sensor}_ema"] = df.groupby('unit_nr')[sensor].transform(
            lambda x: x.ewm(alpha=alpha).mean()
        )
    return df

def apply_savgol(df, sensors, window=15, polyorder=2):
    """S-G Filter 피처 생성"""
    safe_window = window if window % 2 == 1 else window + 1
    for sensor in sensors:
        try:
            df[f"{sensor}_sg"] = df.groupby('unit_nr')[sensor].transform(
                lambda x: savgol_filter(x, window_length=min(safe_window, len(x)) if len(x) > polyorder else len(x), 
                                      polyorder=polyorder) if len(x) > polyorder else x
            )
        except:
            df[f"{sensor}_sg"] = df[sensor]
    return df

def apply_pca(train_df, test_df, sensors):
    """PCA 피처 및 Trend 생성 (여기가 수정됨!)"""
    # print("   [Logic] Applying PCA & Trend...")
    pca_scaler = StandardScaler()
    pca = PCA(n_components=1)

    # 1. PCA Calculation (Train Fit -> Test Transform)
    train_scaled = pca_scaler.fit_transform(train_df[sensors].values)
    train_pc1 = pca.fit_transform(train_scaled)
    train_df['pca_1'] = train_pc1

    test_scaled = pca_scaler.transform(test_df[sensors].values)
    test_pc1 = pca.transform(test_scaled)
    test_df['pca_1'] = test_pc1
    
    # 2. [수정됨] Trend Calculation (누락되었던 부분)
    # pca_1의 변화량(미분값)을 계산합니다.
    for df in [train_df, test_df]:
        df['pca_1_trend'] = df.groupby('unit_nr')['pca_1'].transform(
            lambda x: x.diff().fillna(0)
        )
    
    return train_df, test_df

# ==========================================
# 2. 데이터 준비 (Dynamic Logic)
# ==========================================
def prepare_test_data_dynamic(window_size, feature_cols):
    
    IMPORTANT_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15']
    col_names = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # 1. Raw Data Load
    train_df = pd.read_csv(DATA_DIR / 'raw/train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv(DATA_DIR / 'raw/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    
    ## true rul 로드 후 바로 클리핑
    rul_true = pd.read_csv(DATA_DIR / 'raw/RUL_FD001.txt', sep=r'\s+', header=None).values.flatten()

    ## [중요] -> 평가용 정답지도 학습과 동일하게 125로 제한
    MAX_RUL = 125
    rul_true = np.clip(rul_true, a_min=None, a_max= MAX_RUL)

    # 2. Logic Check
    needs_ema = any("_ema" in col for col in feature_cols)
    needs_sg = any("_sg" in col for col in feature_cols)
    # pca_1이든 pca_1_trend든 'pca'가 들어가면 로직 실행
    needs_pca = any("pca" in col for col in feature_cols) 

    print(f"   [Data] Logic -> EMA: {needs_ema}, SG: {needs_sg}, PCA: {needs_pca}")

    # 3. Apply Logic
    if needs_ema:
        train_df = apply_ema(train_df, IMPORTANT_SENSORS)
        test_df = apply_ema(test_df, IMPORTANT_SENSORS)
    
    if needs_sg:
        train_df = apply_savgol(train_df, IMPORTANT_SENSORS)
        test_df = apply_savgol(test_df, IMPORTANT_SENSORS)
        
    if needs_pca:
        train_df, test_df = apply_pca(train_df, test_df, IMPORTANT_SENSORS)

    # 4. Scaling
    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_cols:
        print(f"   🚨 ERROR: Missing columns: {missing_cols}")
        return None, None

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # 5. Windowing
    X_test_list = []
    y_test_list = []

    for unit_id, group in test_df.groupby('unit_nr'):
        data = group[feature_cols].values
        if len(data) < window_size: continue
        
        X_test_list.append(data[-window_size:])
        y_test_list.append(rul_true[unit_id - 1])

    return torch.tensor(np.array(X_test_list), dtype=torch.float32), torch.tensor(np.array(y_test_list), dtype=torch.float32)

# ==========================================
# 3. 메인 평가 실행
# ==========================================
def evaluate_top_models(top_n=2):
    print(f"🔎 Searching for Top {top_n} models...")
    
    experiment = mlflow.get_experiment_by_name("Turbofan_RUL_Prediction")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=top_n
    )

    for index, run in runs.iterrows():
        run_id = run.run_id
        model_name = run['params.model_type']
        window_size = int(run['params.window_size'])
        
        try:
            feature_cols = ast.literal_eval(run['params.features'])
        except:
            from src.models.model_config import TRAINER_CONFIG
            feature_cols = TRAINER_CONFIG['features']

        print(f"\n[{index+1}/{top_n}] Evaluating {model_name} (ID: {run_id})")
        
        try:
            model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            model.eval()
        except:
            print("   ❌ Load Failed.")
            continue

        X_test, y_true = prepare_test_data_dynamic(window_size, feature_cols)
        
        if X_test is None:
            continue

        with torch.no_grad():
            y_pred = model(X_test)
        
        y_pred_flat = y_pred.numpy().flatten()
        y_true_flat = y_true.numpy().flatten()
        test_rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        print(f"   🏆 Test RMSE: {test_rmse:.4f} (Train RMSE: {run['metrics.train_rmse']:.4f})")

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_rmse", test_rmse)

if __name__ == "__main__":
    evaluate_top_models(top_n=1)