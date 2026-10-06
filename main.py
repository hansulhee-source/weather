import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", layout="wide")

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 평가 및 비교")
st.write("""
서울 기온 데이터를 기반으로 다양한 학습 기간별 선형회귀 모델을 생성하고,
공통 테스트 데이터(최근 20년: 2006~2025)에 대한 예측 성능(MAE, MSE, R²) 및 기울기를 비교합니다.
""")

# ---------------------------------------------------------
# 데이터 로드 및 전처리
# ---------------------------------------------------------
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
    
    # 관측일 300일 이상인 연도만 필터링
    yearly_cleaned = yearly[yearly["관측일수"] >= 300].copy()
    
    # 독립변수 X: 1908년 기준 경과 연수
    yearly_cleaned["X"] = yearly_cleaned["연도"] - 1908
    
    return yearly_cleaned

# 평가 지표 계산 함수 (MAE, MSE, R²)
def evaluate_model(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    
    # R² Score 계산
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return mae, mse, r2

try:
    df_yearly = load_data()

    # ---------------------------------------------------------
    # 데이터셋 분할 (학습 & 테스트)
    # ---------------------------------------------------------
    # 공통 테스트 데이터: 최근 20년 (2006 ~ 2025)
    df_test = df_yearly[(df_yearly["연도"] >= 2006) & (df_yearly["연도"] <= 2025)]
    X_test = df_test["X"].values
    Y_test = df_test["평균기온"].values

    # 모델 1: 전체 데이터 (참고용)
    df_train_full = df_yearly.copy()
    
    # 모델 2: 과거 100년 (1906 ~ 2005)
    df_train_100 = df_yearly[(df_yearly["연도"] >= 1906) & (df_yearly["연도"] <= 2005)]
    
    # 모델 3: 과거 50년 (1956 ~ 2005)
    df_train_50 = df_yearly[(df_yearly["연도"] >= 1956) & (df_yearly["연도"] <= 2005)]

    # ---------------------------------------------------------
    # 모델 학습 및 회귀계수 계산
    # ---------------------------------------------------------
    models = {}
    train_datasets = {
        "전체 데이터": df_train_full,
        "과거 100년 (1906~2005)": df_train_100,
        "과거 50년 (1956~2005)": df_train_50
    }

    for name, df_tr in train_datasets.items():
        X_tr = df_tr["X"].values
        Y_tr = df_tr["평균기온"].values
        
        # 1차 회귀선 계수 구하기 (Y = slope * X + intercept)
        slope, intercept = np.polyfit(X_tr, Y_tr, 1)
        
        # 학습 데이터에 대한 자체 예측 및 평가
        pred_tr = slope * X_tr + intercept
        mae_tr, mse_tr, r2_tr = evaluate_model(Y_tr, pred_tr)
        
        # 테스트 데이터(2006~2025)에 대한 예측 및 평가
        pred_te = slope * X_test + intercept
        mae_te, mse_te, r2_te = evaluate_model(Y_test, pred_te)
        
        models[name] = {
            "slope": slope,
            "intercept": intercept,
            "rate_100y": slope * 100,  # 100년당 상승온도
            "train_n": len(df_tr),
            "train_mae": mae_tr,
            "train_mse": mse_tr,
            "train_r2": r2_tr,
            "test_mae": mae_te,
            "test_mse": mse_te,
            "test_r2": r2_te
        }

    # ---------------------------------------------------------
    # 대시보드 화면 구성
    # ---------------------------------------------------------
    st.subheader("📊 데이터셋 기본 정보")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 분석 연도 수", f"{len(df_yearly)} 년")
    c2.metric("공통 테스트 데이터 수", f"{len(df_test)} 년 (2006~2025)")
    c3.metric("과거 100년 학습 데이터 수", f"{len(df_train_100)} 년")
    c4.metric("과거 50년 학습 데이터 수", f"{len(df_train_50)} 년")

    st.markdown("---")

    # ---------------------------------------------------------
    # 모델 성능 비교 표
    # ---------------------------------------------------------
    st.subheader("⚖️ 학습 기간별 모델 비교 (테스트 데이터 평가: 최근 20년)")
    
    summary_data = []
    for name, m in models.items():
        summary_data.append({
            "학습 모델": name,
            "학습 데이터 수": f"{m['train_n']}개",
            "기울기 (100년당 상승량)": f"{m['rate_100y']:+.2f} °C",
            "테스트 MAE": f"{m['test_mae']:.4f} °C",
            "테스트 MSE": f"{m['test_mse']:.4f}",
            "테스트 R² Score": f"{m['test_r2']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Key Metrics 비교 카드
    st.subheader("🔥 과거 100년 vs 과거 50년 모델 주요 비교")
    col_m1, col_m2 = st.columns(2)
    
    m100 = models["과거 100년 (1906~2005)"]
    m50 = models["과거 50년 (1956~2005)"]

    with col_m1:
        st.write("### 📜 과거 100년 학습 모델 (1906~2005)")
        st.metric("100년당 상승 기온", f"{m100['rate_100y']:+.2f} °C / 100년")
        st.metric("테스트 MAE (평균절대오차)", f"{m100['test_mae']:.4f} °C")
        st.metric("테스트 R² Score", f"{m100['test_r2']:.4f}")

    with col_m2:
        diff_rate = m50['rate_100y'] - m100['rate_100y']
        diff_mae = m50['test_mae'] - m100['test_mae']
        diff_r2 = m50['test_r2'] - m100['test_r2']

        st.write("### ⚡ 과거 50년 학습 모델 (1956~2005)")
        st.metric("100년당 상승 기온", f"{m50['rate_100y']:+.2f} °C / 100년", delta=f"{diff_rate:+.2f} °C (더 가파름)")
        st.metric("테스트 MAE (평균절대오차)", f"{m50['test_mae']:.4f} °C", delta=f"{diff_mae:+.4f} °C", delta_color="inverse")
        st.metric("테스트 R² Score", f"{m50['test_r2']:.4f}", delta=f"{diff_r2:+.4f}")

    st.info("""
    💡 **결과 분석 인사이트**:
    - **기울기 차이**: 과거 50년(1956~2005) 데이터로 학습한 회귀선이 과거 100년 모델보다 기울기가 더 가파릅니다. 이는 20세기 후반으로 갈수록 온난화 속도가 빨라졌음을 보여줍니다.
    - **예측 성능 차이**: 최근 20년(2006~2025)을 예측할 때는 **최근 추세를 더 잘 반영한 과거 50년 학습 모델**이 오차(MAE)가 더 적고 결정계수($R^2$)가 높아 최근 기온을 훨씬 더 정확하게 예측합니다.
    """)

    st.markdown("---")

    # ---------------------------------------------------------
    # Plotly 그래프 시각화
    # ---------------------------------------------------------
    st.subheader("📈 학습 모델별 회귀선 및 테스트 데이터 비교")

    line_years = np.arange(1900, 2101)
    line_x = line_years - 1908

    fig = go.Figure()

    # 1. 학습 데이터 산점도 (1906~2005)
    df_train_past = df_yearly[df_yearly["연도"] <= 2005]
    fig.add_trace(go.Scatter(
        x=df_train_past["연도"],
        y=df_train_past["평균기온"],
        mode="markers",
        name="과거 학습 데이터 (~2005)",
        marker=dict(color="steelblue", size=6, opacity=0.6)
    ))

    # 2. 공통 테스트 데이터 산점도 (2006~2025)
    fig.add_trace(go.Scatter(
        x=df_test["연도"],
        y=df_test["평균기온"],
        mode="markers",
        name="공통 테스트 데이터 (2006~2025)",
        marker=dict(color="crimson", size=9, symbol="diamond")
    ))

    # 3. 전체 데이터 회귀선
    m_full = models["전체 데이터"]
    fig.add_trace(go.Scatter(
        x=line_years,
        y=m_full["slope"] * line_x + m_full["intercept"],
        mode="lines",
        name=f"전체 기간 모델 (+{m_full['rate_100y']:.2f}°C/100년)",
        line=dict(color="gray", width=2, dash="dash")
    ))

    # 4. 과거 100년 회귀선
    fig.add_trace(go.Scatter(
        x=line_years,
        y=m100["slope"] * line_x + m100["intercept"],
        mode="lines",
        name=f"과거 100년 모델 (+{m100['rate_100y']:.2f}°C/100년)",
        line=dict(color="blue", width=2.5)
    ))

    # 5. 과거 50년 회귀선
    fig.add_trace(go.Scatter(
        x=line_years,
        y=m50["slope"] * line_x + m50["intercept"],
        mode="lines",
        name=f"과거 50년 모델 (+{m50['rate_100y']:.2f}°C/100년)",
        line=dict(color="darkorange", width=2.5)
    ))

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="평균 기온 (°C)",
        xaxis=dict(range=[1895, 2030]),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
