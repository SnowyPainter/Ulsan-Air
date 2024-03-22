import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import data_prepare

df = data_prepare.remove_outliers()

# Feature와 Target 분할
X = df.drop(columns=['tmpd'])  # feature
y = df['tmpd']  # target
# 데이터 분할 (학습용 데이터와 테스트용 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 서포트 벡터 머신 회귀 모델 생성 및 학습
svr_model = SVR(kernel='linear')  # 선형 커널 사용
svr_model.fit(X_train, y_train)
# 테스트 데이터로 예측
y_pred = svr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)