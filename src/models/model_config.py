## 개별 모델의 하이퍼파라미터 관리할 모델 하이퍼파라미터 레지스트리# 1. 공통 학습 설정 (Trainer Config)

TRAINER_CONFIG = {
    "batch_size": 256,      # 튜닝 완료된 값
    "epochs": 100,          # 충분히 길게
    "learning_rate": 0.005,
    "patience": 10,         # Early Stopping 용
    "features": [
        'sensor_2_sg', 'sensor_3_sg', 'sensor_4_sg', 'sensor_7_sg',
        'sensor_11_sg', 'sensor_12_sg', 'sensor_15_sg',

        # 새로 추가된 핵심 feature
        'pca_1', # 종합 건강 지수
        'pca_1_trend' # 건강 악화 속도
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
        "window_size": 30, # window len이 길수록 더 긴 과거를 보게 됨
        # "len(hidden_layers)층의 layer 쌓기, 갈수록 뚱뚱해지게 제작"
        "hidden_layers": [32, 64, 128],
        "kernel_size": 3, # 커널 크기 결정 -> 윈도우 크기와 커널 크기를 비례하게..
        "dropout": 0.2 # 레이어 용량과 드롭아웃도 비례하게 설정
    },

    "Transformer": {
        "window_size": 60, # window size가 길수록 사이클을 더 길게 보게 됨
        "d_model": 256, # 모델 용량
        "nhead": 8, # 시야각(데이터를 쳐다보는 눈)
        "num_layers": 4, # 레이어 갯수 -> 추론의 깊이
        "dropout":0.3 # 모델 용량과 드롭아웃 비례하게 설정
    },

    "CNNAttention": {
        "window_size": 40,
        "hidden_dim":128,
        "kernel_size":3,
        "dropout":0.3
    },

    
    "DLinear": {
        "window_size": 50, # DLinear는 긴 시계열을 잘 보므로 30 -> 60으로 늘림 (튜닝 포인트)
        "individual": True # 채널별로 가중치 따로 쓸지 여부
    }
}