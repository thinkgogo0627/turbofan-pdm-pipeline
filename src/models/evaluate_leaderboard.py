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
from sklearn.linear_model import LinearRegression
import ast

# 프로젝트 경로 설정
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"


# [수정 1] GPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Evaluation Device: {device}")

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
def prepare_test_data_dynamic(window_size, feature_cols, min_length_limit=None):
    # min_length_limit: 앙상블 시 데이터 개수를 맞추기 위한 강제 커트라인
    
    IMPORTANT_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15']
    col_names = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # 1. Raw Data Load
    train_df = pd.read_csv(DATA_DIR / 'raw/train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv(DATA_DIR / 'raw/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    
    rul_true = pd.read_csv(DATA_DIR / 'raw/RUL_FD001.txt', sep=r'\s+', header=None).values.flatten()
    MAX_RUL = 125
    rul_true = np.clip(rul_true, a_min=None, a_max=MAX_RUL)

    # 2. Logic Check & Apply
    needs_ema = any("_ema" in col for col in feature_cols)
    needs_sg = any("_sg" in col for col in feature_cols)
    needs_pca = any("pca" in col for col in feature_cols) 

    if needs_ema:
        train_df = apply_ema(train_df, IMPORTANT_SENSORS)
        test_df = apply_ema(test_df, IMPORTANT_SENSORS)
    
    if needs_sg:
        train_df = apply_savgol(train_df, IMPORTANT_SENSORS)
        test_df = apply_savgol(test_df, IMPORTANT_SENSORS)
        
    if needs_pca:
        train_df, test_df = apply_pca(train_df, test_df, IMPORTANT_SENSORS)

    # 4. Scaling
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # 5. Windowing
    X_test_list = []
    y_test_list = []

    # [수정] 커트라인 설정 (없으면 자기 window_size가 커트라인)
    threshold = min_length_limit if min_length_limit is not None else window_size

    for unit_id, group in test_df.groupby('unit_nr'):
        data = group[feature_cols].values
        
        # [핵심 수정] 앙상블 싱크를 맞추기 위해 threshold보다 짧으면 무조건 스킵
        if len(data) < threshold: 
            continue
        
        # 데이터가 충분해도, 모델 입력에는 딱 window_size만큼만 잘라서 넣음 (뒤에서부터)
        X_test_list.append(data[-window_size:])
        y_test_list.append(rul_true[unit_id - 1])

    return torch.tensor(np.array(X_test_list), dtype=torch.float32), torch.tensor(np.array(y_test_list), dtype=torch.float32)


# ==========================================
# 2-1. TTA - MC Dropout
# ==========================================
def predict_with_uncertainty(model, X, n_iter=20):
    """
    MC Dropout: 추론 시에도 Dropout을 켜고 여러 번 예측 후 평균 계산
    """
    model.train() # [중요] eval()이 아니라 train() 모드로 둬야 Dropout이 작동함!
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_iter):
            # 매 반복마다 Dropout이 다르게 터지면서 조금씩 다른 예측값이 나옴
            pred = model(X)
            predictions.append(pred.cpu().numpy().flatten())
            
    # (n_iter, batch_size) -> 평균내서 최종 예측값 도출
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0) # 불확실성(표준편차)도 덤으로 얻음
    
    return mean_pred, std_pred


# ==========================================
# 3. 메인 평가 실행
# ==========================================
def evaluate_top_models(top_n=5):
    print(f"🔎 Searching for Top {top_n} models...")
    
    experiment = mlflow.get_experiment_by_name("Turbofan_RUL_Prediction")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"], # 검증 rmse 기준으로 오름차순 정렬
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
            
            # 모델을 GPU로 이동
            model.to(device)
            model.eval()
        except:
            print("   ❌ Load Failed.")
            continue

        X_test, y_true = prepare_test_data_dynamic(window_size, feature_cols)
        
        if X_test is None:
            continue
        
        # 데이터를 GPU로 이동 및 Shape 맞추기
        # X_test 원본: (Batch, Window, Feature)

        ## CNN / CNNAttention: Conv1d 사용 -> (Batch, Feature, Time) -> Transpose 해야함
        ## Transformer: Linear 레이어 -> (Batch, Time, Feature) -> Transpose 하면 안됨

        X_test = X_test.to(device)
        
        # [Shape 맞추기 로직]
        if "CNN" in model_name:
            # CNN 계열: (Batch, Feature, Window) 형태가 필요함 -> Transpose O
            X_test = X_test.transpose(1, 2)
        elif "Transformer" in model_name or "DLinear" in model_name or "CNNAttention" in model_name:
            # Transformer/DLinear: (Batch, Window, Feature) 형태 유지 -> Transpose X
            pass 
        else:
            X_test = X_test.transpose(1, 2)

        # [수정] 일반 예측 라인(model(X_test)) 삭제하고 바로 MC Dropout 실행
        print(f"   🎲 Applying MC Dropout (n_iter=30)...")
        y_pred_flat, uncertainty = predict_with_uncertainty(model, X_test, n_iter=30)
        
        # CPU로 가져오기 (predict_with_uncertainty 안에서 이미 cpu().numpy() 처리됨)
        y_true_flat = y_true.numpy().flatten()

        test_rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        print(f"   🏆 Test RMSE (MC Dropout): {test_rmse:.4f} (Val RMSE: {run['metrics.val_rmse']:.4f})")

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_rmse_mc_dropout", test_rmse)



# ==========================================
# 4. 앙상블 메인 평가 실행
# ==========================================
def evaluate_ensemble(search_top_n=10, ensemble_top_n=3):
    print(f"🚀 [Ensemble] Scanning Top {search_top_n} models to pick Best {ensemble_top_n}...")
    
    # 실험 설정
    mlflow.set_experiment("Turbofan_RUL_Prediction") 
    
    
    experiment = mlflow.get_experiment_by_name("Turbofan_RUL_Prediction")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"], 
        max_results=search_top_n
    )
    
    # [단계 1] 전체 모델 중 가장 큰 Window Size 찾기 (Synchronization)
    max_window_size_in_pool = 0
    for index, run in runs.iterrows():
        ws = int(run['params.window_size'])
        if ws > max_window_size_in_pool:
            max_window_size_in_pool = ws
    
    print(f"   ⚖️  Enforcing Global Min Length: {max_window_size_in_pool} (to sync all models)")

    candidates = [] 
    y_true_flat = None # 기준 정답지 (가장 긴 윈도우 기준)
    
    # [단계 2] 평가 루프
    for index, run in runs.iterrows():
        run_id = run.run_id
        model_name = run['params.model_type']
        window_size = int(run['params.window_size'])
        
        # 모델 로드
        try:
            model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            model.to(device)
        except:
            continue
            
        try:
            feature_cols = ast.literal_eval(run['params.features'])
        except:
            from src.models.model_config import TRAINER_CONFIG
            feature_cols = TRAINER_CONFIG['features']
            
        # [핵심] min_length_limit에 max_window_size_in_pool을 넣어줍니다.
        # 이렇게 하면 Window 30짜리 모델도 길이가 90인 데이터만 골라서 평가하므로,
        # Window 90짜리 모델과 평가 데이터셋(행 개수)이 똑같아집니다.
        X_test, y_true = prepare_test_data_dynamic(window_size, feature_cols, min_length_limit=max_window_size_in_pool)
        
        X_test = X_test.to(device)
        
        # Shape 맞춤
        if "CNN" in model_name and "CNNAttention" not in model_name:
             X_test = X_test.transpose(1, 2)
        elif "Simple1DCNN" in model_name:
             X_test = X_test.transpose(1, 2)
        
        # MC Dropout 예측
        y_pred, _ = predict_with_uncertainty(model, X_test, n_iter=20)
        
        # 기준 정답지 저장 (한 번만)
        if y_true_flat is None:
            y_true_flat = y_true.numpy().flatten()
            print(f"   ✅ Test Set Size Synced: {len(y_true_flat)} samples")
            
        # 개별 성능 측정
        individual_rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred))
        print(f"  Candidate {index+1} ({model_name} / W={window_size}): RMSE {individual_rmse:.4f}")
        
        candidates.append((individual_rmse, y_pred, model_name))

    # [단계 3] 상위 N개 선정 및 앙상블
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:ensemble_top_n]
    
    print(f"\n✨ Selected Top {ensemble_top_n} Models for Ensemble:")
    selected_preds = []
    for rmse, pred, name in top_candidates:
        print(f"  -> {name} (RMSE: {rmse:.4f})")
        selected_preds.append(pred)
        
    final_pred = np.mean(selected_preds, axis=0)
    final_rmse = np.sqrt(mean_squared_error(y_true_flat, final_pred))
    
    print(f"\n🏆 Final Optimized Ensemble RMSE: {final_rmse:.4f}")

    # [이 부분이 실행되어야 MLflow에 남습니다!]
    with mlflow.start_run(run_name="Ensemble_Final_Top3"):
        mlflow.log_metric("test_rmse", final_rmse)
        mlflow.log_param("method", "Ensemble + MC Dropout")
        mlflow.log_param("models_count", ensemble_top_n)
        
        # 선택된 모델 이름들도 기록
        selected_model_names = [item[2] for item in top_candidates]
        mlflow.log_param("selected_models", str(selected_model_names))
        
        print("📝 Logged Final Score to MLflow UI successfully.") # 이 메시지가 떠야 성공!

    return final_rmse

############################################
# 5. Stacking
#############################################
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import torch
import mlflow
import ast
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# (기존 import 및 device 설정, prepare_test_data_dynamic 등은 위쪽에 있다고 가정)

# ----------------------------------------------------------------
# 1. Validation 데이터 준비 함수 (Sync 기능 포함)
# ----------------------------------------------------------------
def prepare_val_data_synced(window_size, feature_cols, align_max_window=None):
    data_path = PROJECT_DIR / "data/processed/train_FD001_advanced_features.parquet"
    df = pd.read_parquet(data_path)
    
    # RUL Clipping
    MAX_RUL = 125
    df['RUL'] = df['RUL'].clip(upper=MAX_RUL)

    # Split
    unit_ids = df['unit_nr'].unique()
    split_idx = int(len(unit_ids) * 0.8)
    val_units = unit_ids[split_idx:]
    val_df = df[df['unit_nr'].isin(val_units)].copy()

    # Scaling
    scaler = MinMaxScaler()
    val_df[feature_cols] = scaler.fit_transform(val_df[feature_cols])

    X_list, y_list = [], []
    target_sync_len = align_max_window if align_max_window is not None else window_size

    for unit_id, group in val_df.groupby('unit_nr'):
        data = group[feature_cols].values
        target = group['RUL'].values
        
        if len(data) < target_sync_len: continue
        
        # [수정] Transpose 제거! 순수한 (Samples, Window, Feature)로 반환
        windows = sliding_window_view(data, window_shape=window_size, axis=0)
        
        target_windows = target[window_size-1:]
        
        # Sync Logic (Truncate)
        if align_max_window is not None and window_size < align_max_window:
            diff = align_max_window - window_size
            windows = windows[diff:]
            target_windows = target_windows[diff:]
            
        X_list.append(windows)
        y_list.append(target_windows)

    if not X_list: return None, None
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return torch.tensor(X, dtype=torch.float32), y

# ----------------------------------------------------------------
# 2. 스태킹 실행 함수 (명확한 Transpose 분기)
# ----------------------------------------------------------------
def evaluate_linear_blending(top_n=3):
    print(f"🚀 [Stacking] Learning Optimal Weights from Validation Set (Top {top_n})...")
    
    mlflow.set_experiment("Turbofan_RUL_Prediction")
    experiment = mlflow.get_experiment_by_name("Turbofan_RUL_Prediction")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"], 
        max_results=top_n
    )
    
    # [단계 1] Max Window Size 찾기
    max_window_in_pool = 0
    for index, run in runs.iterrows():
        ws = int(run['params.window_size'])
        if ws > max_window_in_pool: max_window_in_pool = ws
            
    print(f"   ⚖️  Syncing Validation Data to Window Size: {max_window_in_pool}")
    
    val_preds_matrix = [] 
    val_y_true = None
    test_preds_matrix = []
    test_y_true = None
    models_loaded = []
    
    for index, run in runs.iterrows():
        run_id = run.run_id
        model_name = run['params.model_type']
        window_size = int(run['params.window_size'])
        
        # 피처 파싱 안전장치
        try:
            feature_cols = ast.literal_eval(run['params.features'])
        except:
            feature_cols = [
                'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15',
                'pca_1', 'pca_1_trend' 
            ]

        print(f"  Load Model {index+1}: {model_name} (W={window_size})")
        
        try:
            model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
            model.eval()
        except:
            print("   ❌ Load Failed.")
            continue
            
        # -------------------------------------------------------
        # A. Validation 예측
        # -------------------------------------------------------
        X_val, y_val = prepare_val_data_synced(window_size, feature_cols, align_max_window=max_window_in_pool)
        if X_val is None: continue
        X_val = X_val.to(device)
        
        # 🔥 [Shape Auto-Correction] 데이터 모양을 보고 강제로 맞춤 🔥
        # 현재 데이터의 마지막 차원이 Window Size(70)인지 Feature(9)인지 확인
        last_dim = X_val.shape[-1]
        
        if "CNN" in model_name and "CNNAttention" not in model_name: 
            # CNN은 (N, F, W)여야 함 -> 마지막이 Window(70)이어야 함
            # 만약 마지막이 Feature(9)라면 -> 뒤집어라
            if last_dim == len(feature_cols): 
                X_val = X_val.transpose(1, 2)
                
        elif "Simple1DCNN" in model_name:
             if last_dim == len(feature_cols): 
                X_val = X_val.transpose(1, 2)
                
        else:
            # Transformer/DLinear 등은 (N, W, F)여야 함 -> 마지막이 Feature(9)여야 함
            # 만약 마지막이 Window(70)라면 -> 뒤집어라 (여기가 에러 원인이었음!)
            if last_dim == window_size: 
                X_val = X_val.transpose(1, 2)

        with torch.no_grad():
            p_val = model(X_val).cpu().numpy().flatten()
            
        val_preds_matrix.append(p_val)
        if val_y_true is None: val_y_true = y_val

        # -------------------------------------------------------
        # B. Test 예측
        # -------------------------------------------------------
        X_test, y_test = prepare_test_data_dynamic(window_size, feature_cols, min_length_limit=max_window_in_pool)
        if X_test is None: continue
        X_test = X_test.to(device)
        
        # Test 데이터도 똑같이 Auto-Correction 적용
        last_dim_test = X_test.shape[-1]
        
        if "CNN" in model_name and "CNNAttention" not in model_name: 
             if last_dim_test == len(feature_cols): X_test = X_test.transpose(1, 2)
        elif "Simple1DCNN" in model_name: 
             if last_dim_test == len(feature_cols): X_test = X_test.transpose(1, 2)
        else:
             # Transformer
             if last_dim_test == window_size: X_test = X_test.transpose(1, 2)

        p_test, _ = predict_with_uncertainty(model, X_test, n_iter=20)
        test_preds_matrix.append(p_test)
        
        if test_y_true is None: test_y_true = y_test.numpy().flatten()
        models_loaded.append(f"{model_name}(W={window_size})")

    # [단계 3] Stacking
    X_meta_train = np.column_stack(val_preds_matrix)
    y_meta_train = val_y_true
    
    meta_model = LinearRegression(positive=True, fit_intercept=False)
    meta_model.fit(X_meta_train, y_meta_train)
    
    weights = meta_model.coef_
    weights = weights / np.sum(weights)
    
    print(f"\n⚖️  Optimal Weights Found: {weights}")
    for name, w in zip(models_loaded, weights):
        print(f"  -> {name}: {w:.4f}")

    # [단계 4] Inference
    X_meta_test = np.column_stack(test_preds_matrix)
    final_pred = np.dot(X_meta_test, weights)
    final_rmse = np.sqrt(mean_squared_error(test_y_true, final_pred))
    
    print(f"\n🏆 Final Stacking RMSE: {final_rmse:.4f}")
    
    with mlflow.start_run(run_name="Stacking_Linear_Blending"):
        mlflow.log_metric("test_rmse", final_rmse)
        mlflow.log_param("weights", str(weights))
        mlflow.log_param("method", "Linear Blending Stacking")
        print("📝 Logged Stacking Score to MLflow UI successfully.")

        
if __name__ == "__main__":
    evaluate_linear_blending(top_n=3)
'''
if __name__ == "__main__":
    # Top 10개를 훑어서 -> 그 중 제일 잘한 3개만 섞어라!
    evaluate_ensemble(search_top_n=10, ensemble_top_n=3)
    '''

'''
if __name__ == "__main__":
    evaluate_top_models(top_n=10)
    '''