import torch
import torch.nn as nn
import math

# ==========================================
# 0. Simple 1D CNN (Baseline)
# ==========================================
class Simple1DCNN(nn.Module):
    def __init__(self, input_dim):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=2) # AdaptiveAvgPool 쓰면 굳이 필요 없음
        self.flatten = nn.Flatten()
        
        # 어떤 길이(Seq_Len)가 들어와도 1개로 압축 (Flatten 편하게 하기 위해)
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.global_pool(x) # (Batch, 32, 1)
        x = self.flatten(x)     # (Batch, 32)
        x = self.fc(x)          # (Batch, 1)
        return x


# ==========================================
# 1. Deep CNN (ResNet Style)
# ==========================================
class DeepCNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, kernel_size= 3, dropout= 0.3):
        super(DeepCNN, self).__init__()

        self.layers = nn.ModuleList() # 여기에 레이어 한 층씩 담기

        # 반복문 돌면서 층 쌓기
        current_dim = input_dim

        for h_dim in hidden_layers:
            # 블록 하나 생성 (Conv -> BN -> ReLU -> Dropout)
            block = nn.Sequential(
                nn.Conv1d(current_dim, h_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(block) # 리스트에 추가
            current_dim = h_dim # 다음 레이어의 입력은 현재 레이어의 출력이 됨

        # 마지막 출력층
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_dim, 1) # 마지막 남은 채널 수 -> 1 (RUL)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.transpose(1, 2)

        # ModuleList에 담긴 레이어들을 순서대로 통과
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)

        return x
        
# ==========================================
# 2. CNN + Attention (SE-Block Style)
# ==========================================
class CNNAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3, dropout=0.2):
        super(CNNAttention, self).__init__()
        
        # 1. Feature Extractor (CNN)
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding='same') # padding='same' 추천
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) # [추가] 과적합 방지턱
        
        # 2. Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # [수정] 표현력 증대
            nn.Tanh(),
            nn.Linear(hidden_dim, 1), # [수정] 정석적인 Attention 구조 (Vector -> Scalar Score)
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.transpose(1, 2)
        
        # CNN
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out) # Dropout 적용
        
        # Attention을 위해 다시 (Batch, Seq, Hidden)으로 전환
        out = out.transpose(1, 2)
        
        # Attention Weights 계산
        weights = self.attention(out)
        
        # Context Vector (가중합)
        context = torch.sum(out * weights, dim=1)
        
        # Prediction
        return self.fc(context)
# ==========================================
# 3. Transformer (Encoder Only)
# ==========================================
class PositionalEncoding(nn.Module): # 상대적 위치 인코딩
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        x = self.embedding(x)  # Input Projection
        x = self.pos_encoder(x) # 위치 정보 주입
        
        x = self.transformer_encoder(x) # (Batch, Seq, d_model)
        
        # 마지막 시점(Last Time Step)만 사용 or 평균 사용
        # 여기서는 평균 사용 (Global Average Pooling)
        x = x.mean(dim=1) 
        
        x = self.fc(x)
        return x
    
# ==========================================
# 4. DLinear (from "Are Transformers Effective for Time Series Forecasting?", AAAI 2023)
# ==========================================

class SeriesDecomp(nn.Module):
    """
    시계열 분해 (Series Decomposition) Block
    - 입력 데이터를 '추세(Trend)'와 '나머지(Seasonal/Residual)'로 분리
    - 방법: Moving Average(이동평균)를 사용하여 추세를 추출
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        # Moving Average를 위한 Average Pooling
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Channel]
        """
        # Padding을 해서 길이를 유지 (앞뒤로 채움)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        # AvgPool은 (Batch, Channel, Seq) 형태를 원함 -> Transpose
        x_pad = x_pad.permute(0, 2, 1)
        trend = self.avg(x_pad)
        trend = trend.permute(0, 2, 1) # 다시 (Batch, Seq, Channel)로 복구
        
        # Seasonal = 원본 - 추세
        seasonal = x - trend
        return seasonal, trend

class DLinear(nn.Module):
    """
    DLinear for RUL Prediction (Regression Ver.)
    - Forecasting이 아니라 Regression이므로, Seq_Len을 1로 압축하는 구조
    """
    def __init__(self, seq_len, input_dim):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        
        # 1. Decomposition (이동평균으로 추세 분리)
        # kernel_size는 윈도우 크기에 따라 조절 (보통 25 정도면 적당)
        self.decomp = SeriesDecomp(kernel_size=25)
        
        # 2. Linear Layers (Time Axis: Seq_Len -> 1)
        # 추세용 Linear와 계절성용 Linear를 따로 둠
        self.linear_trend = nn.Linear(seq_len, 1)
        self.linear_seasonal = nn.Linear(seq_len, 1)
        
        # 3. Final Projection (Channel Axis: Input_Dim -> 1)
        # 채널별로 압축된 정보를 합쳐서 최종 RUL 예측
        self.final_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        
        # Step 1: 분해 (Decomposition)
        seasonal_init, trend_init = self.decomp(x)
        
        # Step 2: Linear Mapping (시간축 압축)
        # Linear를 '시간(Time)' 축에 적용하기 위해 Permute: [Batch, Channel, Seq_Len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # 결과: [Batch, Channel, 1]
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        
        # Step 3: 합체 (Recomposition)
        x = seasonal_output + trend_output
        x = x.squeeze(-1) # [Batch, Channel]
        
        # Step 4: 최종 예측
        x = self.final_fc(x) # [Batch, 1]
        return x