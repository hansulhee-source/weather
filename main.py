import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 기온 데이터 생성 (1906~2025년)
np.random.seed(100)
years = np.arange(1906, 2026)
trend = np.where(
    years <= 1955,
    10.8 + 0.008 * (years - 1906),
    10.8 + 0.008 * (1955 - 1906) + 0.028 * (years - 1955)
)
temps = np.round(trend + np.random.normal(0, 0.45, size=len(years)), 2)
df = pd.DataFrame({'year': years, 'temp': temps})

# 2. 데이터 분할
train_50 = df[(df['year'] >= 1956) & (df['year'] <= 2005)]
train_100 = df[(df['year'] >= 1906) & (df['year'] <= 2005)]
test_20 = df[(df['year'] >= 2006) & (df['year'] <= 2025)]

# 3. 모델 평가 함수
def run_evaluation(train_df, test_df, label):
    X_tr, y_tr = train_df[['year']], train_df['temp']
    X_te, y_te = test_df[['year']], test_df['temp']
    
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    
    pred_te = model.predict(X_te)
    slope_10yr = model.coef_[0] * 10
    
    print(f"[{label}]")
    print(f" - 기울기 (10년당): +{slope_10yr:.3f} ℃")
    print(f" - MAE: {mean_absolute_error(y_te, pred_te):.3f} ℃")
    print(f" - MSE: {mean_squared_error(y_te, pred_te):.3f}")
    print(f" - R² : {r2_score(y_te, pred_te):.3f}\n")

run_evaluation(train_100, test_20, "최근 100년 학습 모델 (1906~2005)")
run_evaluation(train_50, test_20, "최근 50년 학습 모델 (1956~2005)")
