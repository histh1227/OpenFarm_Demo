import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.neighbors import NearestNeighbors
from supabase import create_client, Client
import datetime
import plotly.graph_objects as go # 시각화를 위한 라이브러리 추가

# 초기 설정 및 환경 변수
SUPABASE_URL = "https://qomywlvffwmoyqiacxrf.supabase.co"      
SUPABASE_KEY = "sb_publishable_PlIXJIfv1j04II9lDOKYXA_A36egCtt" 

# 페이지 기본 설정
st.set_page_config(page_title="OpenFarm AI Agent Demo", page_icon="🌱", layout="wide") # 시각화를 위해 넓은 레이아웃 권장

@st.cache_resource
def init_system():
    """모델과 DB 클라이언트를 한 번만 로드하여 캐싱합니다."""
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    try:
        model = YOLO('best.pt')
    except Exception as e:
        model = None
        st.error(f"모델 로드 실패: {e}")
    return supabase, model

supabase, model = init_system()

# 선택 가능한 상태 3개
SCENARIOS = {
    "🌱 상태 A (초기)": {"temp": 24.2, "hum": 44.3, "co2": 419, "img_path": "sample1.jpg"},
    "🌿 상태 B (성장기)": {"temp": 22.71, "hum": 45.06, "co2": 730, "img_path": "sample2.jpg"},
    "🌳 상태 C (수확기)": {"temp": 23.67, "hum": 62.4, "co2": 684, "img_path": "sample3.jpg"}
}

# 메인 함수
def calculate_area(img_path):
    """YOLOv8을 이용해 식물 면적 계산 및 시각화 이미지 생성"""
    if model is None: return 0, None
    results = model(img_path, verbose=False)
    plant_area = 0
    annotated_img = None
    
    for result in results:
        # 픽셀 면적 계산
        if result.masks is not None:
            for mask_xy in result.masks.xy:
                points = np.array(mask_xy, dtype=np.int32)
                h, w = result.orig_img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                plant_area += np.sum(mask == 255)
        
        # YOLOv8 내장 플롯 함수로 바운딩 박스 및 마스크가 그려진 이미지 생성
        annotated_img = result.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
    return plant_area, annotated_img

def run_demo_knn(current_area):
    """KNN 최적 환경 산출 및 산점도용 데이터 반환"""
    try:
        response = supabase.table("model_data").select("*").execute()
        df = pd.DataFrame(response.data)
        
        # DB에 데이터가 충분하지 않을 경우 (기본 추천값)
        if len(df) < 5: 
            return 22.5, 60.0, None
            
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
        df['plant_area'] = pd.to_numeric(df['plant_area'], errors='coerce')
        
        # 단순 성장률(이전 데이터 대비) 계산
        df['growth_rate'] = df['plant_area'].pct_change() * 100
        valid_db = df.dropna(subset=['plant_area', 'temperature', 'humidity', 'growth_rate']).reset_index(drop=True)
        
        if len(valid_db) >= 5:
            knn = NearestNeighbors(n_neighbors=5)
            knn.fit(valid_db[['plant_area']].values)
            distances, indices = knn.kneighbors([[current_area]])
            
            similar_cases = valid_db.iloc[indices[0]]
            best_idx = similar_cases['growth_rate'].idxmax()
            best_case = valid_db.loc[best_idx]
            
            # 시각화를 위해 분석 데이터셋 모음 반환
            plot_data = {
                'df': valid_db,
                'current_area': current_area,
                'neighbor_indices': indices[0],
                'best_index': best_idx
            }
            return round(best_case['temperature'], 1), round(best_case['humidity'], 1), plot_data
        else:
            return 22.5, 60.0, None # 기본값
    except Exception as e:
        st.warning(f"KNN 분석 중 오류 (기본값 반환): {e}")
        return 22.5, 60.0, None

# 웹 UI 구성
st.title("🌱 OpenFarm AI Agent demo")
st.markdown("""
해당 시뮬레이터는 하드웨어 연동 없이 **식물 상호작용형 OpenFarm 자율 제어 시스템**의 핵심 기능(YOLOv8 픽셀 추출 및 환경 추천)을 확인해볼 수 있는 버젼입니다. 
상황을 선택하고 AI Agent의 분석 결과를 확인해보세요.
""")

st.divider()

# 사용자 입력 (시나리오 선택)
st.subheader("1. 현재 상태 선택 (입력 데이터)")
selected_scenario_name = st.radio("테스트할 식물 상태를 선택하세요:", list(SCENARIOS.keys()))
scenario_data = SCENARIOS[selected_scenario_name]

col1, col2 = st.columns([1, 1])

with col1:
    st.info(f"**센서 측정값**\n\n"
            f"🌡️ 현재 온도: {scenario_data['temp']} °C\n\n"
            f"💧 현재 습도: {scenario_data['hum']} %\n\n"
            f"☁️ 이산화탄소: {scenario_data['co2']} ppm")

with col2:
    try:
        # 이미지를 RGB로 변환하여 출력
        img_bgr = cv2.imread(scenario_data['img_path'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="현재 식물 이미지", use_container_width=True)
    except:
        st.error(f"이미지를 찾을 수 없습니다: {scenario_data['img_path']}")

st.divider()

# AI 분석 실행 버튼
if st.button("🚀 AI 자율 분석 및 제어 실행", type="primary", use_container_width=True):
    with st.spinner('YOLOv8 픽셀 추출을 통한 생장 상태 분석 및 KNN 데이터 추론 중...'):
        
        # 픽셀 면적 계산 및 시각화 이미지 획득
        area, annotated_img = calculate_area(scenario_data['img_path'])
        
        # KNN 최적 환경 추천 및 산점도용 데이터 획득
        target_temp, target_hum, plot_data = run_demo_knn(area)
        
        # Supabase 로그 업로드
        log_data = {
            "device_id": "openfarm_demo",
            "temperature": scenario_data['temp'],
            "humidity": scenario_data['hum'],
            "co2_level": scenario_data['co2'],
            "plant_area": float(area),
            "recommended_temp": target_temp,
            "recommended_hum": target_hum,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        try:
            supabase.table("recommend_logs").insert(log_data).execute()
            db_status = "✅ Supabase 로그 저장 성공"
        except Exception as e:
            db_status = f"❌ Supabase 저장 실패: {e}"

    # 결과 출력
    st.success("AI 에이전트 분석이 완료되었습니다!")
    
    st.subheader("2. AI Agent 분석 결과 (출력 데이터)")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="측정된 식물 면적 (YOLOv8)", value=f"{area:,} px")
        st.caption("비전 AI를 통해 식물의 실제 크기를 추출했습니다.")
        
    with res_col2:
        st.metric(label="목표 최적 온도 (KNN)", value=f"{target_temp} °C", delta=f"{round(target_temp - scenario_data['temp'], 1)} °C 조절 필요")
        st.metric(label="목표 최적 습도 (KNN)", value=f"{target_hum} %", delta=f"{round(target_hum - scenario_data['hum'], 1)} % 조절 필요")
        
    st.info(f"**시스템 상태**: AI Agent가 현재 생육 상태({area}px)와 유사했던 과거 최적의 데이터를 바탕으로, 환경 제어기에 목표 설정값을 전송합니다. ({db_status})")
    
    st.divider()

    # AI 시각화 섹션 추가
    st.subheader("3. AI 에이전트 판단 과정 시각화")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**🌱 YOLOv8 객체 인식 및 마스킹 결과**")
        if annotated_img is not None:
            st.image(annotated_img, caption="식물 영역(Mask) 추출 및 바운딩 박스", use_container_width=True)
        else:
            st.info("객체를 인식하지 못했거나 모델을 로드할 수 없습니다.")
            
    with viz_col2:
        st.markdown("**📊 KNN 최적 환경 추천 근거 데이터 (산점도)**")
        if plot_data is not None:
            df = plot_data['df']
            current_area = plot_data['current_area']
            neighbors = df.iloc[plot_data['neighbor_indices']]
            best_case = df.loc[plot_data['best_index']]
            
            fig = go.Figure()

            # 1. 과거 전체 데이터
            fig.add_trace(go.Scatter(
                x=df['plant_area'], y=df['growth_rate'],
                mode='markers', name='과거 데이터',
                marker=dict(color='lightgray', size=8)
            ))

            # 2. KNN으로 찾은 유사 크기 데이터 5개
            fig.add_trace(go.Scatter(
                x=neighbors['plant_area'], y=neighbors['growth_rate'],
                mode='markers', name='유사 크기 (KNN Top 5)',
                marker=dict(color='royalblue', size=12, symbol='circle-open', line=dict(width=2))
            ))

            # 3. 그 중 성장률이 가장 높은 최적의 데이터
            fig.add_trace(go.Scatter(
                x=[best_case['plant_area']], y=[best_case['growth_rate']],
                mode='markers', name='최적의 성장 케이스',
                marker=dict(color='red', size=15, symbol='star')
            ))

            # 4. 현재 측정된 면적 기준선
            fig.add_vline(x=current_area, line_dash="dash", line_color="green", 
                          annotation_text=f"현재 측정 면적 ({int(current_area):,} px)")

            fig.update_layout(
                xaxis_title="식물 픽셀 면적 (px)",
                yaxis_title="다음 구간까지의 성장률 (%)",
                legend=dict(orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5),
                margin=dict(l=0, r=0, t=30, b=60)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("데이터베이스에 시각화할 충분한 과거 데이터가 없습니다. (기본 추천값 적용됨)")
