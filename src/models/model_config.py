## 개별 모델의 하이퍼파라미터 관리할 모델 하이퍼파라미터 레지스트리# 1. 공통 학습 설정 (Trainer Config)

TRAINER_CONFIG = {
    "batch_size": 256,      # 튜닝 완료된 값
    "epochs": 100,          # 충분히 길게
    "learning_rate": 0.005,
    "patience": 10,         # Early Stopping 용
    "features": [
        'sensor_2_ema', 'sensor_3_ema', 'sensor_4_ema', 'sensor_7_ema',
        'sensor_11_ema', 'sensor_12_ema', 'sensor_15_ema'
    ]
}

# 2. 모델별 구조 설정 (Model Specific Configs)
# -> 여기서 모델 내부 파라미터를 구조화합니다.
MODEL_CONFIGS = {
    "Simple1DCNN": {
        "window_size": 30,
        "kernel_size": 3,
        "filters": 32
    },
    "DeepCNN": {
        "window_size": 30,
        "hidden_dim": 64,
        "layers": 3,
        "dropout": 0.3
    },
    "Transformer": {
        "window_size": 30, # Transformer는 윈도우가 좀 더 길어도 됨
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2
    },
    "DLinear": {
        "window_size": 60, # DLinear는 긴 시계열을 잘 보므로 30 -> 60으로 늘림 (튜닝 포인트)
        "individual": False # 채널별로 가중치 따로 쓸지 여부
    }
}