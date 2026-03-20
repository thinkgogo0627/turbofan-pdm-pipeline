import streamlit as st
import requests
import pandas as pd
import time
import numpy as np

# 1. 웹 페이지 기본 세팅
st.set_page_config(page_title="Pit Wall Dashboard", page_icon="🏎️", layout="wide")
st.title("Turbofan RUL 실시간 모니터링")
st.markdown("---")

# FastAPI 서버 주소
API_URL = "http://localhost:8000/predict"

# 2. 사이드바 제어반
st.sidebar.header("Control Panel")
engine_id = st.sidebar.selectbox("모니터링할 Engine ID", [1, 3, 5, 12, 24])
start_btn = st.sidebar.button("🚀 스트리밍 시작")

if start_btn:
    st.sidebar.success(f"Engine #{engine_id} 데이터 연결 완료!")
    
    # 지표와 차트를 그릴 빈 공간(Placeholder) 마련
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # 그래프를 그리기 위한 데이터 저장소
    cycle_history = []
    rul_history = []
    
    # 3. 실시간 스트리밍 시뮬레이션 (비행 사이클 진행)
    for current_cycle in range(1, 100):
        # ⚠️ [주의] 실제 환경에서는 Pandas로 NASA CSV 파일을 읽어와서 
        # 해당 엔진의 최근 Window_size(예: 70)만큼의 데이터를 슬라이싱해서 보내야 합니다.
        # 여기서는 서버 통신 테스트를 위해 임의의 형태(Dummy)를 만들어 쏩니다.
        
        # Pydantic schemas.py 규격에 맞춘 JSON 페이로드 생성
        feature_columns = [
            "time_cycles", "sensor_2", "sensor_3", "sensor_4", 
            "sensor_7", "sensor_11", "sensor_12", "sensor_15"
        ]
        
        # 2. 70사이클 x 8개 피처 크기의 랜덤 소수점 데이터 생성
        random_data_array = np.random.rand(300, 8)
        
        # 3. 각 행(row)마다 이름표를 붙이고, time_cycles는 정수로 강제 변환!
        formatted_data = []
        for i, row in enumerate(random_data_array):
            row_dict = dict(zip(feature_columns, row))
            row_dict["time_cycles"] = int(current_cycle + i)  # 👈 핵심! 사이클을 정수(int)로 바꿔줍니다
            formatted_data.append(row_dict)

        # 4. 완벽한 규격으로 payload 완성!
        payload = {
            "unit_id": engine_id,
            "cycle": current_cycle,
            "data": formatted_data 
        }
        
        try:
            # 백엔드 엔진에 예측 요청 발사!
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                # 결과값 파싱
                pred_rul = response.json().get("predicted_rul", 60)
                
                # 가상의 RUL 하락 시뮬레이션 (시간이 갈수록 줄어들게 임의 조정)
                # 실제 데이터 연동 시 이 줄은 지우시면 됩니다.
                pred_rul = max(0, 60 - (current_cycle * 0.5) + np.random.normal(0, 2))
                
                cycle_history.append(current_cycle)
                rul_history.append(pred_rul)
                
                # 4. 실시간 UI 업데이트
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Cycle", current_cycle)
                    
                    # RUL이 30 이하면 경고색(빨간색)으로 표시되도록 조치
                    status = "🚨 WARNING" if pred_rul <= 30 else "🟢 NORMAL"
                    col2.metric("Predicted RUL", f"{pred_rul:.1f}")
                    col3.metric("Engine Status", status)
                
                # 선 그래프 그리기
                df_chart = pd.DataFrame({"Cycle": cycle_history, "RUL": rul_history}).set_index("Cycle")
                chart_placeholder.line_chart(df_chart, color="#ff4b4b" if pred_rul <= 30 else "#00a8ff")
                
            else:
                st.error(f"서버 에러: {response.text}")
                break
                
        except Exception as e:
            st.error(f"백엔드 서버에 연결할 수 없습니다. 도커 컨테이너가 켜져 있는지 확인하세요! Error: {e}")
            break
            
        # 0.5초 대기 후 다음 사이클 진행 (실시간 느낌 연출)
        time.sleep(0.5)

    st.success("해당 엔진의 비행 사이클이 종료되었습니다.")