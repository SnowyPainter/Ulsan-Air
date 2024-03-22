import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def get_raw():
    '''
    tmpd: tempature, pm10tmean2: Fine particulate, o3tmean2: ozone, no2tmean2: no2
    '''
    df = pd.read_csv('./chicago_air_pollution.csv')
    df.drop(["city", "dptp", "pm25tmean2"], axis=1, inplace=True)
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index(["date"], inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df

def draw_graph(df):
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.title('Normalized Environmental Data')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def remove_outliers():
    raw = get_raw()
    m = raw.mean()
    s = raw.std()
    df = (raw - m) / s
    # 오존, 온도 | 미세먼지, no2

    train_x = df[['pm10tmean2', 'tmpd', 'pm10tmean2', 'no2tmean2']].values

    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.05)
    y_pred1 = lof.fit_predict(train_x)
    outliers_indexes = np.where(y_pred1 == -1)[0]
    outliers_values = df.iloc[outliers_indexes]

    #이상치가 도출되었을 때의 상관관계 분석
    print(outliers_values.corr())
    #오존과 기온은 상관관계가 큼을 확인

    #이상치 제거
    df = df.drop(df.index[outliers_indexes])

    return df