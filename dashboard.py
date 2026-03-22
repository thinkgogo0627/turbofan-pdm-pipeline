import streamlit as st
import pandas as pd
import time
import numpy as np

st.set_page_config(page_title="Pit Wall Dashboard", page_icon="🏎️", layout="wide")
st.title(" Turbofan RUL & SPC 통계적 공정 관리")
st.markdown("---")

st.sidebar.header("Control Panel")
engine_id = st.sidebar.selectbox("모니터링할 Engine ID", [1, 3, 5, 12, 24])
start_btn = st.sidebar.button("🚀 스트리밍 시작")

if start_btn:
    st.sidebar.success(f"Engine #{engine_id} 텔레메트리 연결 완료!")
    
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    alert_placeholder = st.empty() # 경고 메시지용 공간 추가
    
    # 📌 통계적 공정 관리(SPC) 파라미터 세팅
    HEALTHY_MEAN_RUL = 60.0    # 건강한 상태의 평균 RUL
    SIGMA = 4.0                # 임의로 설정한 RUL 변동성 표준편차
    UCL = HEALTHY_MEAN_RUL + (3 * SIGMA)  # 상한선 (72)
    LCL = HEALTHY_MEAN_RUL - (3 * SIGMA)  # 하한선 (48)
    
    # 그래프 데이터 저장소
    history = {"Cycle": [], "Predicted RUL": [], "UCL (+3σ)": [], "LCL (-3σ)": []}
    
    for current_cycle in range(1, 100):
        # 1. 랜덤 센서 데이터 생성 (입구 컷 회피용 포맷)
        feature_columns = ["time_cycles", "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]
        random_data_array = np.random.rand(70, 8)
        
        formatted_data = []
        for i, row in enumerate(random_data_array):
            row_dict = dict(zip(feature_columns, row))
            row_dict["time_cycles"] = int(current_cycle + i) 
            formatted_data.append(row_dict)

        # (FastAPI 통신은 생략하고 임의의 RUL 하락을 시뮬레이션합니다)
        # 60에서 시작해서 사이클당 0.5씩 떨어지며 노이즈(오차)가 섞인 형태
        pred_rul = max(0, 60 - (current_cycle * 0.5) + np.random.normal(0, 1.5))
        
        # 데이터 누적
        history["Cycle"].append(current_cycle)
        history["Predicted RUL"].append(pred_rul)
        history["UCL (+3σ)"].append(UCL)
        history["LCL (-3σ)"].append(LCL)
        
        # 2. UI 업데이트
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Cycle", current_cycle)
            col2.metric("Predicted RUL", f"{pred_rul:.1f}")
            
            # 🚨 3-Sigma 룰 검사 (LCL 이탈 확인)
            if pred_rul < LCL:
                col3.metric("SPC Status", "🚨 LCL 이탈 (열화 진행)")
                alert_placeholder.error(f"⚠️ [경고] Cycle {current_cycle}: RUL이 정상 통계 범위(-3σ)를 이탈하여 본격적인 수명 단축이 시작되었습니다!")
            else:
                col3.metric("SPC Status", "🟢 In Control")
                alert_placeholder.empty()

        # 3. 3개의 선(예측값, UCL, LCL)을 동시에 그리기
        df_chart = pd.DataFrame(history).set_index("Cycle")
        chart_placeholder.line_chart(
            df_chart, 
            color=["#00a8ff", "#ff4b4b", "#ff4b4b"] # 예측선은 파란색, 기준선은 빨간색
        )
            
        time.sleep(0.3)

    st.success("해당 엔진의 비행 사이클이 종료되었습니다.")