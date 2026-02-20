import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schema import PredictRequest, PredictResponse
from app.inference import load_artifacts, predict_rul, ensemble_models

# 최신 FastAPI 방식: 서버 시작/종료 시 실행될 로직을 관리하는 라이프사이클 매니저
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 서버 켜질 때 가중치와 전처리기 메모리 로드
    print("🏎️ [Pit Wall] Starting Power Unit initialization...")
    load_artifacts()
    print("🟢 [Pit Wall] All systems go. Ready for telemetry data.")
    yield
    # Shutdown: 서버 꺼질 때 정리 로직 (필요시)
    print("🏁 [Pit Wall] Shutting down Power Unit...")

# FastAPI 앱 초기화
app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="MGU-K(Validation) + ICE(Inference) = PU(FastAPI)",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Turbofan RUL Prediction Power Unit is running."}

@app.post("/predict", response_model=PredictResponse)
def predict_engine_rul(request: PredictRequest):
    try:
        start_time = time.time()
        
        # 1. MGU-K(schemas.py)를 통과한 데이터를 ICE(inference.py)로 전달
        rul_value = predict_rul(request.data)
        
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"⏱️ [Telemetry] Prediction completed in {inference_time:.4f} seconds.")

        # 2. 결과를 스키마에 맞춰서 반환
        return PredictResponse(
            predicted_rul=rul_value,
            ensemble_models_used=len(ensemble_models)
        )
    
    except Exception as e:
        print(f"❌ [Error] Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")