import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울의 과거 기온 데이터를 분석하고, 회귀 모델을 통해 특정 연도의 예상 기온을 확인합니다.")

# 데이터 불러오기 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 처리 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 선택
    df = df[df["연도"] <= 2025]
    
    # 연도별 관측일수 및 평균기온 집계
    yearly = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일이 300일 이상인 연도만 정제
    yearly_cleaned = yearly[yearly["관측일수"] >= 300].copy()
    
    # 독립변수 X: 1908년부터 경과한 연수
    yearly_cleaned["X"] = yearly_cleaned["연도"] - 1908
    
    return yearly_cleaned

try:
    df_yearly = load_data()

    # 데이터 통계 정보 계산
    n_years = len(df_yearly)
    min_year = int(df_yearly["연도"].min())
    max_year = int(df_yearly["연도"].max())

    # 선형 회귀 계수 계산 (Y = slope * X + intercept)
    X = df_yearly["X"].values
    Y = df_yearly["평균기온"].values
    slope, intercept = np.polyfit(X, Y, 1)

    # 상관계수 계산
    corr = np.corrcoef(X, Y)[0, 1]

    # 기본 정보 출력
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("분석에 사용된 연도 수", f"{n_years} 개")
    col_info2.metric("시작 연도", f"{min_year} 년")
    col_info3.metric("끝 연도", f"{max_year} 년")
    col_info4.metric("상관계수 (r)", f"{corr:.4f}")

    st.markdown("---")

    # 슬라이더 및 예측값 표시 영역
    st.subheader("🔮 연도별 예상 기온 예측")
    selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

    # 선택된 연도의 예상 기온 계산
    pred_x = selected_year - 1908
    predicted_temp = slope * pred_x + intercept

    st.metric(
        label=f"{selected_year}년 예상 연평균 기온",
        value=f"{predicted_temp:.2f} °C"
    )

    st.markdown("---")

    # Plotly 시각화
    st.subheader("📈 서울 연평균 기온 및 회귀 직선")

    # 회귀선 그리기용 연도 범위 (1900~2100)
    line_years = np.arange(1900, 2101)
    line_x = line_years - 1908
    line_y = slope * line_x + intercept

    fig = go.Figure()

    # 1. 실제 관측 데이터 산점도
    fig.add_trace(go.Scatter(
        x=df_yearly["연도"],
        y=df_yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(color="royalblue", size=7)
    ))

    # 2. 회귀 직선
    fig.add_trace(go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="firebrick", width=2)
    ))

    # 3. 슬라이더 선택 지점 강조 표시
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers+text",
        name="선택된 연도 예측값",
        text=[f"{selected_year}년: {predicted_temp:.2f}°C"],
        textposition="top center",
        marker=dict(color="orange", size=14, symbol="star")
    ))

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="평균 기온 (°C)",
        xaxis=dict(range=[1895, 2105]),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
