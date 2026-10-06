import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# 한글 폰트 설정 (기본 폰트 지정)
plt.rcParams['font.sans-serif'] = ['NanumGothic', 'Malgun Gothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------------
# 1. 데이터 생성/로드 함수 (캐싱 처리)
# -------------------------------------------------------------------
@st.cache_data
def generate_temperature_data():
    np.random.seed(100)
    years = np.arange(1906, 2026) # 1906년 ~ 2025년 (120년간)
    
    # 온난화 가속 추세 반영 (1955년 이후 기울기 가파라짐)
    trend = np.where(
        years <= 1955,
        10.8 + 0.008 * (years - 1906),
        10.8 + 0.008 * (1955 - 1906) + 0.028 * (years - 1955)
    )
    # 연도별 기후 변동 노이즈 추가
    temps = np.round(trend + np.random.normal(0, 0.45, size=len(years)), 2)
    return pd.DataFrame({'year': years, 'temp': temps})

# -------------------------------------------------------------------
# 2. 메인 실행 영역
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="연평균 기온 선형회귀 분석", layout="wide")
    st.title("🌡️ 연평균 기온 선형회귀 모델 비교 분석")
    st.caption("학습 데이터 기간(최근 100년 vs 최근 50년)에 따른 미래(최근 20년) 예측 성능 비교")

    try:
        df = generate_temperature_data()

        # 데이터 세트 분할
        train_100 = df[(df['year'] >= 1906) & (df['year'] <= 2005)]
        train_50  = df[(df['year'] >= 1956) & (df['year'] <= 2005)]
        test_20   = df[(df['year'] >= 2006) & (df['year'] <= 2025)]
        full_data = df.copy()

        # 모델 학습 함수
        def fit_model(train_df):
            X = train_df[['year']].values
            y = train_df['temp'].values
            model = LinearRegression()
            model.fit(X, y)
            return model

        model_full = fit_model(full_data)
        model_100  = fit_model(train_100)
        model_50   = fit_model(train_50)

        # 테스트 데이터(2006~2025) 평가 함수
        def evaluate_on_test(model, test_df):
            X_test = test_df[['year']].values
            y_test = test_df['temp'].values
            y_pred = model.predict(X_test)
            
            slope = model.coef_[0]
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return slope, mae, mse, r2

        slope_f, mae_f, mse_f, r2_f = evaluate_on_test(model_full, test_20)
        slope_100, mae_100, mse_100, r2_100 = evaluate_on_test(model_100, test_20)
        slope_50, mae_50, mse_50, r2_50 = evaluate_on_test(model_50, test_20)

        # -------------------------------------------------------------------
        # 3. Streamlit 화면 출력 - 요약 대시보드
        # -------------------------------------------------------------------
        st.subheader("📊 모델 평가 지표 요약 (테스트: 2006~2025년)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 데이터 모델 (120년)", f"기울기: +{slope_f*10:.3f}℃/10년", f"MAE: {mae_f:.3f}℃")
        with col2:
            st.metric("최근 100년 모델 (1906~2005)", f"기울기: +{slope_100*10:.3f}℃/10년", f"MAE: {mae_100:.3f}℃", delta_color="inverse")
        with col3:
            st.metric("최근 50년 모델 (1956~2005)", f"기울기: +{slope_50*10:.3f}℃/10년", f"MAE: {mae_50:.3f}℃")

        # 세부 요약 표
        summary_df = pd.DataFrame({
            "모델 구분": ["전체 데이터 (1906~2025)", "최근 100년 (1906~2005)", "최근 50년 (1956~2005)"],
            "기울기 (℃/10년)": [f"+{slope_f*10:.3f}", f"+{slope_100*10:.3f}", f"+{slope_50*10:.3f}"],
            "테스트 MAE": [f"{mae_f:.3f}", f"{mae_100:.3f}", f"{mae_50:.3f}"],
            "테스트 MSE": [f"{mse_f:.3f}", f"{mse_100:.3f}", f"{mse_50:.3f}"],
            "테스트 R²": [f"{r2_f:.3f}", f"{r2_100:.3f}", f"{r2_50:.3f}"]
        })
        st.dataframe(summary_df, use_container_width=True)

        # -------------------------------------------------------------------
        # 4. Matplotlib/Seaborn 차트 시각화
        # -------------------------------------------------------------------
        st.subheader("📈 회귀선 및 테스트 데이터 추세 시각화")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 실제 데이터 점 그래프
        sns.scatterplot(data=df, x='year', y='temp', hue=(df['year'] >= 2006), 
                        palette={False: '#1f77b4', True: '#e377c2'}, ax=ax, s=30, alpha=0.7)
        
        # 예측 구간용 X 축 데이터 생성 (1906~2025)
        x_all = np.arange(1906, 202
