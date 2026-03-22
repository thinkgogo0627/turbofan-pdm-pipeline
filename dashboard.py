import streamlit as st
import pandas as pd
import numpy as np
import time
import requests

st.set_page_config(page_title="Turbofan RUL Dashboard", page_icon="⚙️", layout="wide")
st.title("⚙️ Turbofan Engine RUL & SPC Monitoring System")
st.markdown("---")

# 1. 데이터 로드 및 전처리 함수
@st.cache_data 
def load_real_data(file_path):
    columns = ["unit_id", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)
        return df
    except FileNotFoundError:
        return None

# 2. 분석 대상 데이터셋 메타데이터 정의
datasets = [
    {"name": "FD001 (단일 조건)", "path": "data/raw/test_FD001.txt"},
    {"name": "FD002 (다중 조건)", "path": "data/raw/test_FD002.txt"},
    {"name": "FD003 (다중 고장)", "path": "data/raw/test_FD003.txt"},
    {"name": "FD004 (복합 조건)", "path": "data/raw/test_FD004.txt"}
]

# 3. 동적 탭 생성
tab_objects = st.tabs([ds["name"] for ds in datasets])

# 4. For 루프를 이용한 UI 및 비즈니스 로직 동적 할당
for idx, (tab, ds_info) in enumerate(zip(tab_objects, datasets)):
    with tab:
        st.subheader(f"Dataset: {ds_info['name']}")
        
        df = load_real_data(ds_info["path"])
        
        if df is not None:
            available_engines = df["unit_id"].unique().tolist()
            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                # 위젯 Key에 idx를 부여하여 탭 간 충돌 방지
                engine_id = st.selectbox("Engine ID 선택", available_engines, key=f"select_{idx}")
                start_btn = st.button("데이터 스트리밍 시작", key=f"btn_{idx}")
                
            with col_b:
                metrics_placeholder = st.empty()
                chart_placeholder = st.empty()
                alert_placeholder = st.empty()

            if start_btn:
                st.success(f"Engine #{engine_id} 실시간 모니터링을 시작합니다.")
                
                engine_data = df[df["unit_id"] == engine_id].copy()
                
                HEALTHY_MEAN_RUL = 60.0    
                SIGMA = 4.0                
                UCL = HEALTHY_MEAN_RUL + (3 * SIGMA)  
                LCL = HEALTHY_MEAN_RUL - (3 * SIGMA)  
                
                history = {"Cycle": [], "Predicted RUL": [], "UCL (+3σ)": [], "LCL (-3σ)": []}
                total_cycles = engine_data["time_cycles"].max()
                window_size = 70
                
                if total_cycles < window_size:
                    st.error(f"데이터 부족: 최소 필요 윈도우 사이즈({window_size}) 미달. (현재: {total_cycles} cycles)")
                else:
                    for current_cycle in range(window_size, total_cycles + 1):
                        window_df = engine_data[(engine_data["time_cycles"] > current_cycle - window_size) & 
                                                (engine_data["time_cycles"] <= current_cycle)]
                        
                        required_cols = ["time_cycles", "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]
                        payload_data = window_df[required_cols].to_dict(orient="records")
                        
                        payload = {
                            "unit_id": int(engine_id),
                            "cycle": int(current_cycle),
                            "data": payload_data
                        }
                        
                        start_time = time.time()
                        try:
                            response = requests.post("http://localhost:8000/predict", json=payload)
                            latency = (time.time() - start_time) * 1000 

                            if response.status_code == 200:
                                pred_rul = response.json().get("predicted_rul", 0)
                                
                                history["Cycle"].append(current_cycle)
                                history["Predicted RUL"].append(pred_rul)
                                history["UCL (+3σ)"].append(UCL)
                                history["LCL (-3σ)"].append(LCL)
                                
                                with metrics_placeholder.container():
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.metric("Current Cycle", current_cycle)
                                    c2.metric("Predicted RUL", f"{pred_rul:.1f}")
                                    
                                    if pred_rul < LCL:
                                        c3.metric("SPC Status", "🚨 LCL 이탈")
                                        alert_placeholder.error(f"⚠️ [System Alert] Cycle {current_cycle}: RUL 예측값이 정상 하한선(-3σ)을 이탈하여 열화가 진행 중입니다.")
                                    else:
                                        c3.metric("SPC Status", "🟢 Normal")
                                        alert_placeholder.empty()

                                    c4.metric("API Latency", f"{latency:.1f} ms")

                                df_chart = pd.DataFrame(history).set_index("Cycle")
                                chart_placeholder.line_chart(df_chart, color=["#00a8ff", "#ff4b4b", "#ff4b4b"])
                                
                            else:
                                st.error(f"서버 에러 발생: HTTP {response.status_code}")
                                break
                        except Exception as e:
                            st.error(f"백엔드 연결 실패: API 서버 상태를 확인하십시오. ({e})")
                            break
                            
                        time.sleep(0.05) # 빠른 렌더링을 위해 대기 시간 축소
                        
                    with st.expander(f"🛠️ API Network Inspector (백엔드 통신 로그 - {ds_info['name']})"):
                        st.write("⬆️ **최종 Request Payload (to FastAPI):**")
                        st.json({"endpoint": "/predict", "payload_size": f"{len(payload['data'])} rows"})
                        st.write("⬇️ **최종 Response Data (from FastAPI):**")
                        if 'response' in locals() and response.status_code == 200:
                            st.json(response.json())
                            
                    st.success("데이터 스트리밍 및 분석이 완료되었습니다.")
        else:
            st.warning(f"데이터 파일을 찾을 수 없습니다: {ds_info['path']}")