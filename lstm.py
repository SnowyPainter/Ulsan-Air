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

# LSTM에 맞게 데이터 변환
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 3
X_lstm, y_lstm = create_dataset(X, y, TIME_STEPS)

split_index = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:split_index], X_lstm[split_index:]
y_train, y_test = y_lstm[:split_index], y_lstm[split_index:]

model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)
mse = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error:", mse)

#0.541432003990383 SVM - Regression
#0.46861958503723145 LSTM