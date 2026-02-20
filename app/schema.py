from pydantic import BaseModel, Field, validator
from typing import List

## FastAPI 에 올릴 때, 클라이언트의 호출에 대해서 데이터의 무결성을 검증하는 스키마


# 1타임스텝(1초)에 들어오는 센서 데이터 1줄의 형태
class SensorData(BaseModel):
    time_cycles: int = Field(default=1, description="비행 사이클(시간)")
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_11: float
    sensor_12: float
    sensor_15: float

# 클라이언트가 예측을 요청할 때 보내는 전체 데이터 뭉치 (List)
class PredictRequest(BaseModel):
    data: List[SensorData] = Field(..., description="시계열 센서 데이터 (최소 75 타임스텝 이상 필요)")

    @validator('data')
    def check_length(cls, v):
        # 앙상블 모델 중 가장 긴 윈도우 사이즈가 75라고 가정 (ensemble_meta.json 기준)
        if len(v) < 75:
            raise ValueError(f"데이터 길이가 너무 짧습니다. (현재 {len(v)}개, 최소 75개 필요)")
        return v

# 서버가 응답할 결과물의 형태
class PredictResponse(BaseModel):
    predicted_rul: float = Field(..., description="예측된 잔여 수명(RUL)")
    ensemble_models_used: int = Field(..., description="투표에 참여한 앙상블 모델 개수")