import pandas as pd
import numpy as np
from pathlib import Path
import os

# ==========================================
# 1. 설정 및 경로 (Settings)
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "train_FD001.txt"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# 실험으로 확정한 하이퍼파라미터
EMA_SPAN = 7
IMPORTANT_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

# ==========================================
# 2. 핵심 로직 함수 (Logic)
# ==========================================

def load_raw_data():
    """Raw 텍스트 데이터를 읽어옵니다."""
    cols = ['unit_nr', 'time_cycles', 'os_1', 'os_2', 'os_3'] + [f'sensor_{i}' for i in range(1, 22)]
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}. Run ingest_data.py first.")
    
    df = pd.read_csv(RAW_DATA_PATH, sep='\s+', header=None, names=cols)
    print(f"[Load] Raw data loaded. Shape: {df.shape}")
    return df

def calculate_rul(df):
    """RUL(잔여 수명) 라벨 생성"""
    # Max Cycle - Current Cycle
    rul = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    df['RUL'] = rul
    return df

def create_variations(df, sensor_col, span):
    """
    [Feature Engineering Core]
    EMA(노이즈 제거) -> Diff(변화량) -> Std(불확실성) 파생 변수 생성
    """
    # 1. Base: EMA (Span=7)
    ema_col = f"{sensor_col}_ema"
    df[ema_col] = df.groupby('unit_nr')[sensor_col].transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )
    
    # 2. Variation A: Diff (변화 속도)
    # EMA가 1초전보다 얼마나 변했는가?
    df[f"{sensor_col}_diff"] = df.groupby('unit_nr')[ema_col].transform(
        lambda x: x.diff().fillna(0)
    )
    
    # 3. Variation B: Std (최근 흔들림 정도)
    # 최근 7초간 진동폭이 얼마나 컸는가?
    df[f"{sensor_col}_std"] = df.groupby('unit_nr')[sensor_col].transform(
        lambda x: x.rolling(window=span, min_periods=1).std().fillna(0)
    )
    return df

def process_pipeline():
    print("[Start] Building Feature Store...")
    
    # 1. Load
    df = load_raw_data()
    
    # 2. RUL Labeling
    df = calculate_rul(df)
    
    # 3. Feature Engineering (Variations)
    print(f"[Process] Applying EMA(span={EMA_SPAN}) and Creating Variations...")
    for sensor in IMPORTANT_SENSORS:
        df = create_variations(df, sensor, EMA_SPAN)
        
    # 4. Cleaning (불필요한 컬럼 제거 - 선택 사항)
    # 일단은 Feature Store니까 원본 센서값도 남겨두고, 파생변수도 다 저장합니다.
    # 나중에 학습할 때 골라 쓰면 됩니다.
    
    # 5. Save to Parquet
    if not PROCESSED_DIR.exists():
        os.makedirs(PROCESSED_DIR)
        
    save_path = PROCESSED_DIR / "train_FD001_features.parquet"
    
    # Parquet 저장 (압축 사용)
    df.to_parquet(save_path, index=False, engine='pyarrow', compression='snappy')
    
    print("-" * 30)
    print(f"[Done] Feature Store Created at: {save_path}")
    print(f"Final Data Shape: {df.shape}")
    print(f"Columns Example: {list(df.columns[:5])} ... {list(df.columns[-3:])}")
    print("-" * 30)

if __name__ == "__main__":
    process_pipeline()