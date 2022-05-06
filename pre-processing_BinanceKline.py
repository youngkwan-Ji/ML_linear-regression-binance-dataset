import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

batch_size = 32
look_back = 3

def create_LSTM_dataset_X(data):
    data_arr = data
    x_arr = data
    x_arr = np.reshape(x_arr, (1, x_arr.shape[0], x_arr.shape[1]))

    return x_arr

def create_LSTM_dataset_Y(data):
    data_arr = data
    y_arr = []

    for i in range(len(data_arr) - look_back):
        y_arr.append(data_arr[i + look_back, 0:data_arr.shape[1]])

    y_arr = np.array(y_arr)
    y_arr = np.reshape(y_arr, (y_arr.shape[0], y_arr.shape[1]))

    return y_arr

input_str = sys.argv[1]

pre_data_json = json.loads(input_str)
pre_data_df = pd.DataFrame(pre_data_json)
pre_data_df.index = pre_data_df['date']
del pre_data_df['last']
del pre_data_df['date']
pre_data_df = pre_data_df.astype('float')

dataset_df = pd.DataFrame(pre_data_df.diff())
dataset_df = dataset_df.dropna()

x_data = dataset_df[['high', 'low', 'close', 'volume', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseAssetVolume',
                    'takerBuyQuoteAssetVolume']]
y_data = dataset_df[['open','high','low']]

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaler.fit(x_data)
y_scaler.fit(y_data)
x_train_scaled = x_scaler.transform(x_data)

x_train_scaled = create_LSTM_dataset_X(x_train_scaled)

model = tf.keras.models.load_model("services/ML/models/LSTM_Binance-Kline_20220504.h5")
y_predict_scaled = model.predict(x_train_scaled)
y_predict = y_scaler.inverse_transform(y_predict_scaled)

last_data = pre_data_df.iloc[len(pre_data_df)-1]

predict_date = pre_data_df.index[3] + 60000
predict_open = last_data['open'] + y_predict[0][0]
predict_high = last_data['high'] + y_predict[0][1]
predict_low = last_data['low'] + y_predict[0][2]

result = pd.Series({'date' : predict_date
                        ,'open': predict_open
                       ,'high' : predict_high
                       ,'low' :predict_low})

print(result.to_json())