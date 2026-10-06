import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울의 과거 기온 데이터를 분석하고, 회귀 모델을 통해 특정 연도의 예상 기온을 확인합니다.")

# 데이터 불러오기 및 전처리 함수
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

    # 데이터 기본 정보
    n_years = len(df_yearly)
    min_year = int(df_yearly["연도"].min())
    max_year = int(df_yearly["연도"].max())

    # ---------------------------------------------------------
    # 1. 전체 기간 회귀 분석
    # ---------------------------------------------------------
    X_full = df_yearly["X"].values
    Y_full = df_yearly["평균기온"].values
    slope_full, intercept_full = np.polyfit(X_full, Y_full, 1)
    corr_full = np.corrcoef(X_full, Y_full)[0, 1]
    rate_100y_full = slope_full * 100  # 100년당 상승 온도

    # ---------------------------------------------------------
    # 2. 최근 20년 데이터 회귀 분석 (최대 연도 기준 최근 20개 연도)
    # ---------------------------------------------------------
    df_recent20 = df_yearly[df_yearly["연도"] >= (max_year - 19)].copy()
    X_recent = df_recent20["X"].values
    Y_recent = df_recent20["평균기온"].values
    slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)
    corr_recent = np.corrcoef(X_recent, Y_recent)[0, 1]
    rate_100y_recent = slope_recent * 100  # 100년당 상승 온도

    # 기본 데이터 통계 정보
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("분석에 사용된 연도 수", f"{n_years} 개")
    col_info2.metric("시작 연도", f"{min_year} 년")
    col_info3.metric("끝 연도", f"{max_year} 년")

    st.markdown("---")

    # ---------------------------------------------------------
    # 100년당 기온 상승량 비교 표시
    # ---------------------------------------------------------
    st.subheader("🔥 100년당 기온 상승량 비교")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.metric(
            label=f"🌐 전체 기간 ({min_year}~{max_year}) 상승률",
            value=f"{rate_100y_full:+.2f} °C / 100년",
            delta=f"상관계수 r = {corr_full:.4f}"
        )
    with col_m2:
        diff_rate = rate_100y_recent - rate_100y_full
        st.metric(
            label=f"⚡ 최근 20년 ({max_year-19}~{max_year}) 상승률",
            value=f"{rate_100y_recent:+.2f} °C / 100년",
            delta=f"전체 대비 {diff_rate:+.2f} °C (r = {corr_recent:.4f})"
        )

    st.markdown("---")

    # ---------------------------------------------------------
    # 슬라이더 및 연도별 예측 기온 비교
    # ---------------------------------------------------------
    st.subheader("🔮 연도별 예상 기온 예측 비교")
    selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

    pred_x = selected_year - 1908
    pred_temp_full = slope_full * pred_x + intercept_full
    pred_temp_recent = slope_recent * pred_x + intercept_recent

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric(
            label=f"{selected_year}년 예상 기온 (전체 기간 모델)",
            value=f"{pred_temp_full:.2f} °C"
        )
    with col_p2:
        st.metric(
            label=f"{selected_year}년 예상 기온 (최근 20년 모델)",
            value=f"{pred_temp_recent:.2f} °C",
            delta=f"차이 {pred_temp_recent - pred_temp_full:+.2f} °C"
        )

    st.markdown("---")

    # ---------------------------------------------------------
    # Plotly 시각화 (두 회귀선 비교)
    # ---------------------------------------------------------
    st.subheader("📈 서울 연평균 기온 및 회귀 직선 비교")

    line_years = np.arange(1900, 2101)
    line_x = line_years - 1908
    
    line_y_full = slope_full * line_x + intercept_full
    line_y_recent = slope_recent * line_x + intercept_recent

    fig = go.Figure()

    # 1. 실제 관측 데이터 산점도
    fig.add_trace(go.Scatter(
        x=df_yearly["연도"],
        y=df_yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(color="royalblue", size=7, opacity=0.7)
    ))

    # 2. 전체 기간 회귀 직선
    fig.add_trace(go.Scatter(
        x=line_years,
        y=line_y_full,
        mode="lines",
        name=f"전체 기간 추세 (+{rate_100y_full:.2f}°C/100년)",
        line=dict(color="firebrick", width=2.5)
    ))

    # 3. 최근 20년 회귀 직선
    fig.add_trace(go.Scatter(
        x=line_years,
        y=line_y_recent,
        mode="lines",
        name=f"최근 20년 추세 (+{rate_100y_recent:.2f}°C/100년)",
        line=dict(color="darkorange", width=2.5, dash="dash")
    ))

    # 4. 슬라이더 선택 지점 강조 표시 (전체 모델)
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[pred_temp_full],
        mode="markers+text",
        name="선택 연도 (전체 모델)",
        text=[f"{pred_temp_full:.2f}°C"],
        textposition="top center",
        marker=dict(color="firebrick", size=12, symbol="star")
    ))

    # 5. 슬라이더 선택 지점 강조 표시 (최근 20년 모델)
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[pred_temp_recent],
        mode="markers+text",
        name="선택 연도 (최근 20년 모델)",
        text=[f"{pred_temp_recent:.2f}°C"],
        textposition="bottom center",
        marker=dict(color="darkorange", size=12, symbol="diamond")
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
