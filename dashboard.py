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
    # NASA CMAPSS 데이터 로드 (Header 없음, 공백 구분)
    columns = ["unit_id", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)
        return df
    except FileNotFoundError:
        return None

# 2. 데이터셋 도메인별 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["FD001 (단일 조건)", "FD002 (다중 조건)", "FD003 (다중 고장)", "FD004 (복합 조건)"])

# 3. FD001 데이터 스트리밍 로직
with tab1:
    st.subheader("Dataset: FD001 (Standard Conditions)")
    
    # 데이터 파일 경로 설정
    data_path = "data/raw/test_FD001.txt" 
    df_fd001 = load_real_data(data_path)
    
    if df_fd001 is not None:
        # 고유 Unit ID 목록 추출
        available_engines = df_fd001["unit_id"].unique().tolist()
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            engine_id = st.selectbox("Engine ID 선택", available_engines, key="fd001_select")
            start_btn = st.button("데이터 스트리밍 시작", key="fd001_btn")
            
        with col_b:
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            alert_placeholder = st.empty()

        if start_btn:
            st.success(f"Engine #{engine_id} 실시간 모니터링을 시작합니다.")
            
            engine_data = df_fd001[df_fd001["unit_id"] == engine_id].copy()
            
            # SPC (통계적 공정 관리) 파라미터 초기화
            HEALTHY_MEAN_RUL = 60.0    
            SIGMA = 4.0                
            UCL = HEALTHY_MEAN_RUL + (3 * SIGMA)  
            LCL = HEALTHY_MEAN_RUL - (3 * SIGMA)  
            
            history = {"Cycle": [], "Predicted RUL": [], "UCL (+3σ)": [], "LCL (-3σ)": []}
            total_cycles = engine_data["time_cycles"].max()
            window_size = 70
            
            # 최소 윈도우 사이즈 검증
            if total_cycles < window_size:
                st.error(f"데이터 부족: 최소 필요 윈도우 사이즈({window_size}) 미달. (현재: {total_cycles} cycles)")
            else:
                for current_cycle in range(window_size, total_cycles + 1):
                    # Sliding Window 데이터 추출
                    window_df = engine_data[(engine_data["time_cycles"] > current_cycle - window_size) & 
                                            (engine_data["time_cycles"] <= current_cycle)]
                    
                    required_cols = ["time_cycles", "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]
                    payload_data = window_df[required_cols].to_dict(orient="records")
                    
                    payload = {
                        "unit_id": int(engine_id),
                        "cycle": int(current_cycle),
                        "data": payload_data
                    }
                    
                    # API 통신 및 응답 지연 시간(Latency) 측정
                    start_time = time.time()
                    try:
                        response = requests.post("http://localhost:8000/predict", json=payload)
                        latency = (time.time() - start_time) * 1000 

                        if response.status_code == 200:
                            pred_rul = response.json().get("predicted_rul", 0)
                            
                            # 시계열 데이터 누적
                            history["Cycle"].append(current_cycle)
                            history["Predicted RUL"].append(pred_rul)
                            history["UCL (+3σ)"].append(UCL)
                            history["LCL (-3σ)"].append(LCL)
                            
                            # 실시간 지표(Metrics) 업데이트
                            with metrics_placeholder.container():
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Current Cycle", current_cycle)
                                c2.metric("Predicted RUL", f"{pred_rul:.1f}")
                                
                                # 통계적 하한선(LCL) 이탈 감지 로직
                                if pred_rul < LCL:
                                    c3.metric("SPC Status", "🚨 LCL 이탈")
                                    alert_placeholder.error(f"⚠️ [System Alert] Cycle {current_cycle}: RUL 예측값이 정상 하한선(-3σ)을 이탈하여 열화가 진행 중입니다.")
                                else:
                                    c3.metric("SPC Status", "🟢 Normal")
                                    alert_placeholder.empty()

                                c4.metric("API Latency", f"{latency:.1f} ms")

                            # 다중 선 그래프 시각화
                            df_chart = pd.DataFrame(history).set_index("Cycle")
                            chart_placeholder.line_chart(df_chart, color=["#00a8ff", "#ff4b4b", "#ff4b4b"])
                            
                        else:
                            st.error(f"서버 에러 발생: HTTP {response.status_code}")
                            break
                    except Exception as e:
                        st.error(f"백엔드 연결 실패: API 서버 상태를 확인하십시오. ({e})")
                        break
                        
                    time.sleep(0.1) # UI 업데이트 간격 조절
                    
                # 스트리밍 종료 후 Network Inspector 표시
                with st.expander("🛠️ API Network Inspector (백엔드 통신 로그)"):
                    st.write("⬆️ **최종 Request Payload (to FastAPI):**")
                    st.json({"endpoint": "/predict", "payload_size": f"{len(payload['data'])} rows"})
                    st.write("⬇️ **최종 Response Data (from FastAPI):**")
                    if 'response' in locals() and response.status_code == 200:
                        st.json(response.json())
                        
                st.success("데이터 스트리밍 및 분석이 완료되었습니다.")
    else:
        st.warning(f"데이터 파일을 찾을 수 없습니다: {data_path}")

# 4. FD002 ~ FD004 탭 구조
with tab2:
    st.subheader("Dataset: FD002 (Multiple Conditions)")
    
    # 데이터 파일 경로 설정
    data_path = "data/raw/test_FD002.txt" 
    df_fd002 = load_real_data(data_path)
    
    if df_fd002 is not None:
        # 고유 Unit ID 목록 추출
        available_engines = df_fd002["unit_id"].unique().tolist()
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            engine_id = st.selectbox("Engine ID 선택", available_engines, key="fd001_select")
            start_btn = st.button("데이터 스트리밍 시작", key="fd001_btn")
            
        with col_b:
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            alert_placeholder = st.empty()

        if start_btn:
            st.success(f"Engine #{engine_id} 실시간 모니터링을 시작합니다.")
            
            engine_data = df_fd002[df_fd002["unit_id"] == engine_id].copy()
            
            # SPC (통계적 공정 관리) 파라미터 초기화
            HEALTHY_MEAN_RUL = 60.0    
            SIGMA = 4.0                
            UCL = HEALTHY_MEAN_RUL + (3 * SIGMA)  
            LCL = HEALTHY_MEAN_RUL - (3 * SIGMA)  
            
            history = {"Cycle": [], "Predicted RUL": [], "UCL (+3σ)": [], "LCL (-3σ)": []}
            total_cycles = engine_data["time_cycles"].max()
            window_size = 70
            
            # 최소 윈도우 사이즈 검증
            if total_cycles < window_size:
                st.error(f"데이터 부족: 최소 필요 윈도우 사이즈({window_size}) 미달. (현재: {total_cycles} cycles)")
            else:
                for current_cycle in range(window_size, total_cycles + 1):
                    # Sliding Window 데이터 추출
                    window_df = engine_data[(engine_data["time_cycles"] > current_cycle - window_size) & 
                                            (engine_data["time_cycles"] <= current_cycle)]
                    
                    required_cols = ["time_cycles", "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]
                    payload_data = window_df[required_cols].to_dict(orient="records")
                    
                    payload = {
                        "unit_id": int(engine_id),
                        "cycle": int(current_cycle),
                        "data": payload_data
                    }
                    
                    # API 통신 및 응답 지연 시간(Latency) 측정
                    start_time = time.time()
                    try:
                        response = requests.post("http://localhost:8000/predict", json=payload)
                        latency = (time.time() - start_time) * 1000 

                        if response.status_code == 200:
                            pred_rul = response.json().get("predicted_rul", 0)
                            
                            # 시계열 데이터 누적
                            history["Cycle"].append(current_cycle)
                            history["Predicted RUL"].append(pred_rul)
                            history["UCL (+3σ)"].append(UCL)
                            history["LCL (-3σ)"].append(LCL)
                            
                            # 실시간 지표(Metrics) 업데이트
                            with metrics_placeholder.container():
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Current Cycle", current_cycle)
                                c2.metric("Predicted RUL", f"{pred_rul:.1f}")
                                
                                # 통계적 하한선(LCL) 이탈 감지 로직
                                if pred_rul < LCL:
                                    c3.metric("SPC Status", "🚨 LCL 이탈")
                                    alert_placeholder.error(f"⚠️ [System Alert] Cycle {current_cycle}: RUL 예측값이 정상 하한선(-3σ)을 이탈하여 열화가 진행 중입니다.")
                                else:
                                    c3.metric("SPC Status", "🟢 Normal")
                                    alert_placeholder.empty()

                                c4.metric("API Latency", f"{latency:.1f} ms")

                            # 다중 선 그래프 시각화
                            df_chart = pd.DataFrame(history).set_index("Cycle")
                            chart_placeholder.line_chart(df_chart, color=["#00a8ff", "#ff4b4b", "#ff4b4b"])
                            
                        else:
                            st.error(f"서버 에러 발생: HTTP {response.status_code}")
                            break
                    except Exception as e:
                        st.error(f"백엔드 연결 실패: API 서버 상태를 확인하십시오. ({e})")
                        break
                        
                    time.sleep(0.1) # UI 업데이트 간격 조절
                    
                # 스트리밍 종료 후 Network Inspector 표시
                with st.expander("🛠️ API Network Inspector (백엔드 통신 로그)"):
                    st.write("⬆️ **최종 Request Payload (to FastAPI):**")
                    st.json({"endpoint": "/predict", "payload_size": f"{len(payload['data'])} rows"})
                    st.write("⬇️ **최종 Response Data (from FastAPI):**")
                    if 'response' in locals() and response.status_code == 200:
                        st.json(response.json())
                        
                st.success("데이터 스트리밍 및 분석이 완료되었습니다.")
    else:
        st.warning(f"데이터 파일을 찾을 수 없습니다: {data_path}")



with tab3:
    st.subheader("Dataset: FD003 (Multiple Anomaly")
    
    # 데이터 파일 경로 설정
    data_path = "data/raw/test_FD003.txt" 
    df_fd003 = load_real_data(data_path)
    
    if df_fd003 is not None:
        # 고유 Unit ID 목록 추출
        available_engines = df_fd003["unit_id"].unique().tolist()
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            engine_id = st.selectbox("Engine ID 선택", available_engines, key="fd001_select")
            start_btn = st.button("데이터 스트리밍 시작", key="fd001_btn")
            
        with col_b:
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            alert_placeholder = st.empty()

        if start_btn:
            st.success(f"Engine #{engine_id} 실시간 모니터링을 시작합니다.")
            
            engine_data = df_fd003[df_fd003["unit_id"] == engine_id].copy()
            
            # SPC (통계적 공정 관리) 파라미터 초기화
            HEALTHY_MEAN_RUL = 60.0    
            SIGMA = 4.0                
            UCL = HEALTHY_MEAN_RUL + (3 * SIGMA)  
            LCL = HEALTHY_MEAN_RUL - (3 * SIGMA)  
            
            history = {"Cycle": [], "Predicted RUL": [], "UCL (+3σ)": [], "LCL (-3σ)": []}
            total_cycles = engine_data["time_cycles"].max()
            window_size = 70
            
            # 최소 윈도우 사이즈 검증
            if total_cycles < window_size:
                st.error(f"데이터 부족: 최소 필요 윈도우 사이즈({window_size}) 미달. (현재: {total_cycles} cycles)")
            else:
                for current_cycle in range(window_size, total_cycles + 1):
                    # Sliding Window 데이터 추출
                    window_df = engine_data[(engine_data["time_cycles"] > current_cycle - window_size) & 
                                            (engine_data["time_cycles"] <= current_cycle)]
                    
                    required_cols = ["time_cycles", "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]
                    payload_data = window_df[required_cols].to_dict(orient="records")
                    
                    payload = {
                        "unit_id": int(engine_id),
                        "cycle": int(current_cycle),
                        "data": payload_data
                    }
                    
                    # API 통신 및 응답 지연 시간(Latency) 측정
                    start_time = time.time()
                    try:
                        response = requests.post("http://localhost:8000/predict", json=payload)
                        latency = (time.time() - start_time) * 1000 

                        if response.status_code == 200:
                            pred_rul = response.json().get("predicted_rul", 0)
                            
                            # 시계열 데이터 누적
                            history["Cycle"].append(current_cycle)
                            history["Predicted RUL"].append(pred_rul)
                            history["UCL (+3σ)"].append(UCL)
                            history["LCL (-3σ)"].append(LCL)
                            
                            # 실시간 지표(Metrics) 업데이트
                            with metrics_placeholder.container():
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Current Cycle", current_cycle)
                                c2.metric("Predicted RUL", f"{pred_rul:.1f}")
                                
                                # 통계적 하한선(LCL) 이탈 감지 로직
                                if pred_rul < LCL:
                                    c3.metric("SPC Status", "🚨 LCL 이탈")
                                    alert_placeholder.error(f"⚠️ [System Alert] Cycle {current_cycle}: RUL 예측값이 정상 하한선(-3σ)을 이탈하여 열화가 진행 중입니다.")
                                else:
                                    c3.metric("SPC Status", "🟢 Normal")
                                    alert_placeholder.empty()

                                c4.metric("API Latency", f"{latency:.1f} ms")

                            # 다중 선 그래프 시각화
                            df_chart = pd.DataFrame(history).set_index("Cycle")
                            chart_placeholder.line_chart(df_chart, color=["#00a8ff", "#ff4b4b", "#ff4b4b"])
                            
                        else:
                            st.error(f"서버 에러 발생: HTTP {response.status_code}")
                            break
                    except Exception as e:
                        st.error(f"백엔드 연결 실패: API 서버 상태를 확인하십시오. ({e})")
                        break
                        
                    time.sleep(0.1) # UI 업데이트 간격 조절
                    
                # 스트리밍 종료 후 Network Inspector 표시
                with st.expander("🛠️ API Network Inspector (백엔드 통신 로그)"):
                    st.write("⬆️ **최종 Request Payload (to FastAPI):**")
                    st.json({"endpoint": "/predict", "payload_size": f"{len(payload['data'])} rows"})
                    st.write("⬇️ **최종 Response Data (from FastAPI):**")
                    if 'response' in locals() and response.status_code == 200:
                        st.json(response.json())
                        
                st.success("데이터 스트리밍 및 분석이 완료되었습니다.")
    else:
        st.warning(f"데이터 파일을 찾을 수 없습니다: {data_path}")