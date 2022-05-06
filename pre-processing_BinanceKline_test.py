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
    # x_arr = []

    # for i in range(len(data_arr) - look_back):
    #     x_arr.append(data_arr[i:i + look_back, 0:data_arr.shape[1]])
    #
    # x_arr = np.array(x_arr)
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


# bvpValueList=hp.get_data('./BVP.csv')
# input_str = '[{"date":1651642680000,"open":"38067.27000000","high":"38075.69000000","low":"38067.26000000","close":"38073.03000000","volume":"16.76564000","quoteAssetVolume":"638255.46280500","numberOfTrades":270,"takerBuyBaseAssetVolume":"11.54539000","takerBuyQuoteAssetVolume":"439521.51641940","last":true},{"date":1651642740000,"open":"38073.03000000","high":"38074.50000000","low":"38068.19000000","close":"38070.21000000","volume":"10.70706000","quoteAssetVolume":"407633.67938780","numberOfTrades":270,"takerBuyBaseAssetVolume":"5.06924000","takerBuyQuoteAssetVolume":"192987.51788510","last":true},{"date":1651642800000,"open":"38070.22000000","high":"38070.22000000","low":"38061.84000000","close":"38063.29000000","volume":"17.30979000","quoteAssetVolume":"658920.32592720","numberOfTrades":301,"takerBuyBaseAssetVolume":"7.70152000","takerBuyQuoteAssetVolume":"293160.60845750","last":true},{"date":1651642860000,"open":"38063.29000000","high":"38067.23000000","low":"38054.53000000","close":"38058.05000000","volume":"19.08960000","quoteAssetVolume":"726573.12901630","numberOfTrades":338,"takerBuyBaseAssetVolume":"12.36333000","takerBuyQuoteAssetVolume":"470547.90585570","last":true},{"date":1651642920000,"open":"38058.05000000","high":"38071.15000000","low":"38046.18000000","close":"38049.31000000","volume":"52.06582000","quoteAssetVolume":"1981586.09921830","numberOfTrades":646,"takerBuyBaseAssetVolume":"36.25770000","takerBuyQuoteAssetVolume":"1379996.07647780","last":true}]'
input_str = '[{"date":1651642680000,"open":"38067.27000000","high":"38075.69000000","low":"38067.26000000","close":"38073.03000000","volume":"16.76564000","quoteAssetVolume":"638255.46280500","numberOfTrades":270,"takerBuyBaseAssetVolume":"11.54539000","takerBuyQuoteAssetVolume":"439521.51641940","last":true},{"date":1651642740000,"open":"38073.03000000","high":"38074.50000000","low":"38068.19000000","close":"38070.21000000","volume":"10.70706000","quoteAssetVolume":"407633.67938780","numberOfTrades":270,"takerBuyBaseAssetVolume":"5.06924000","takerBuyQuoteAssetVolume":"192987.51788510","last":true},{"date":1651642800000,"open":"38070.22000000","high":"38070.22000000","low":"38061.84000000","close":"38063.29000000","volume":"17.30979000","quoteAssetVolume":"658920.32592720","numberOfTrades":301,"takerBuyBaseAssetVolume":"7.70152000","takerBuyQuoteAssetVolume":"293160.60845750","last":true},{"date":1651642860000,"open":"38063.29000000","high":"38067.23000000","low":"38054.53000000","close":"38058.05000000","volume":"19.08960000","quoteAssetVolume":"726573.12901630","numberOfTrades":338,"takerBuyBaseAssetVolume":"12.36333000","takerBuyQuoteAssetVolume":"470547.90585570","last":true}]'


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
# y_train_scaled = y_scaler.transform(y_data)

# print(x_scaler.inverse_transform(x_train_scaled))

x_train_scaled = create_LSTM_dataset_X(x_train_scaled)
# y_train_scaled = create_LSTM_dataset_Y(y_train_scaled)

model = tf.keras.models.load_model("resource/model/LSTM_Binance-Kline_20220504.h5")
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

# print(pre_data_df['close'])
print(result.to_json())
# last_data['open'] = last_data['open'] + 7
#
# print(last_data['open'])
# print(y_predict)
# print(x_scaler.inverse_transform(x_train_scaled[0]))
# print(x_train_scaled)


#
# with open(sys.argv[1]) as prev_data_list:
#     hrvFeatures = []
#     jsonList = json.load(json_file)
#     bvpDf = pd.DataFrame(jsonList)
#     bvpTimestampList = np.array(bvpDf.timestamp)
#     bvpValueList = hp.preprocessing.scale_data(np.array(bvpDf.dataValue), lower=-300, upper=300)
#
#     fs = 64
#     sec = 60
#
#     while bvpValueList.size >= fs * sec:
#         minuteList = bvpValueList[0:fs * sec]
#
#         working_data, measures = hp.process(minuteList, fs, calc_freq=True, freq_method='welch')
#         result = measures.copy()
#         result['startTime'] = bvpTimestampList[0]
#         result['endTime'] = bvpTimestampList[fs * sec]
#
#         hrvFeatures.append(result)
#         bvpValueList = np.delete(bvpValueList, np.arange(fs))
#         bvpTimestampList = np.delete(bvpTimestampList, np.arange(fs))
#
# print(json.dumps(hrvFeatures))
