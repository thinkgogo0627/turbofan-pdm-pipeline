# src/features/advanced_features.py

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AdvancedFeatureEngineer:
    def __init__(self, sensors):
        self.sensors = sensors
        self.pca = PCA(n_components=1) # 1차원으로 압축 (Health Index)
        self.scaler = StandardScaler()

    def apply_savgol(self, df, window=15, polyorder=2):
        """
        [논문 기반 전략 1] Savitzky-Golay Filter
        - EMA의 단점(신호 지연/Lag)을 해결.
        - 파형의 피크(Peak)를 뭉개지 않고 노이즈만 제거함.
        """
        print(f"[Advanced] Applying Savitzky-Golay Filter (win={window})...")
        for sensor in self.sensors:
            # 윈도우 크기는 홀수여야 함
            safe_window = window if window % 2 == 1 else window + 1
            
            # 각 엔진(unit_nr)별로 필터 적용
            df[f"{sensor}_sg"] = df.groupby('unit_nr')[sensor].transform(
                lambda x: savgol_filter(x, window_length=safe_window, polyorder=polyorder)
            )
        return df

    def apply_pca_fusion(self, df):
        """
        [논문 기반 전략 2] PCA-based Health Indicator
        - 14개 센서가 제각각 움직이는 걸 하나의 '종합 건강 지수'로 합침.
        - 고장 시점 예측에 매우 강력한 Feature.
        """
        print("[Advanced] Creating PCA Health Indicator...")
        
        # 1. 센서 데이터만 추출 및 스케일링
        sensor_data = df[self.sensors].values
        scaled_data = self.scaler.fit_transform(sensor_data)
        
        # 2. PCA 적용 (PC1 추출)
        pc1 = self.pca.fit_transform(scaled_data)
        
        # 3. Feature 추가
        df['pca_1'] = pc1
        
        # 4. (추가) PC1의 변화 속도(Trend)도 정보로 활용
        df['pca_1_trend'] = df.groupby('unit_nr')['pca_1'].transform(
            lambda x: x.diff().fillna(0)
        )
        return df

    def apply_kurtosis(self, df, window=20):
        """
        [논문 기반 전략 3] Rolling Kurtosis (첨도)
        - 평균값 변화 전에 발생하는 '튀는 신호(Impulse)'를 감지.
        - 초기 고장 징후 포착용.
        """
        print("[Advanced] Calculating Rolling Kurtosis...")
        for sensor in self.sensors:
            # rolling().apply()는 느릴 수 있으니 필요한 센서 몇 개만 적용해도 됨
            df[f"{sensor}_kurt"] = df.groupby('unit_nr')[sensor].transform(
                lambda x: x.rolling(window=window).apply(kurtosis, raw=True).fillna(0)
            )
        return df