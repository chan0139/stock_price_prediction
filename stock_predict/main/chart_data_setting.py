from django.conf import settings
import numpy as np
import FinanceDataReader as fdr
import pandas as pd
def get_RSI_14(data):
    RSI_list = []
    for i in range(15, len(data)):  # 15행 종가부터 시작
        close = list(data.iloc[i - 14 : i + 1]['Close']) # [23665, 23572, 23676, ...]
        positive = []
        negative = []
        for j in range(14):
            diff = close[j + 1] - close[j]
            if diff >= 0:
                positive.append(diff)
            else:
                negative.append(diff)

        AU = np.sum(positive) / 13
        AD = abs(np.sum(negative) / 13)
        RSI = AU / (AU + AD)
        RSI_list.append(RSI)

    while len(RSI_list) != len(data):
        RSI_list.insert(0, 0)

    return RSI_list

def stocastic_k(data):   # 14 days
    null_list = []
    for i in range(len(data)):
        calculate_low = np.array(data['Low'][i - 13 : i + 1])
        calculate_high = np.array(data['High'][i - 13 : i + 1])
        if str(calculate_low.mean()) == 'nan':
            continue
        else:
            today = data.iloc[i]['Close']
            mini = calculate_low.min()
            high = calculate_high.max()
            null_list.append((today - mini) / (high - mini))

    while len(null_list) != len(data):
        null_list.insert(0, 0)

    return null_list


def Bollinger(data):    # 20 , 2
    null_list = []
    for i in range(len(data)):
        cal_list = data['Close'][i - 19 : i + 1]
        high = cal_list.mean() + np.std(cal_list) * 2
        low = cal_list.mean() - np.std(cal_list) * 2
        position = (data['Close'][i] - low) /  (high - low)
        null_list.append(position)
    return null_list

def last_day(code, year):
    day = str(fdr.DataReader(str(code), str(year), str(year + 1)).index[-1])[:10]
    print(day)

    return day

def start_day(code, year):
    day = str(fdr.DataReader(str(code), str(year), str(year + 1)).index[0])[:10]
    print(day)

    return day

def last_day_month(code, year, month):
    day = str(fdr.DataReader(str(code), str(year) + '.' + str(month), str(year) + '.' + str(month + 1)).index[-1])[:10]

    return day
def start_day_month(code, year, month):
    day = str(fdr.DataReader(str(code), str(year) + '.' + str(month), str(year) + '.' + str(month + 1)).index[0])[:10]

    return day

def MACD_cat(data):
    null_list = []
    for i in range(1, len(data)):
        if (data['MACD'][i] > 0) & (data['MACD'][i - 1] < 0):
            null_list.append(1)
        else:
            null_list.append(0)
    null_list.insert(0, 0)
    return null_list

def all(data):
    data['RSI'] = get_RSI_14(data)
    data['STOCASTIC_K'] = stocastic_k(data)
    data['STOCASTIC_D'] = data['STOCASTIC_K'].ewm(span = 5).mean()  # 5일
    data['Bollinger'] = Bollinger(data)
    data['MACD'] = data['Close'].ewm(span = 12).mean() - data['Close'].ewm(span = 26).mean()
    data['MACD_SIGNAL'] = data['MACD'].ewm(span = 9).mean()
    data['MACD_cat'] = MACD_cat(data)
    data['Change+'] = list((data['Change'] > 0)[1 : len(data)].astype(int)) + [0]

    data['RSI_delta'] = data.RSI.diff().fillna(0)
#     data['K_delta'] = data.STOCASTIC_K.diff().fillna(0)
    data['D_delta'] = data.STOCASTIC_D.diff().fillna(0)
    data['sto_diff'] = data['STOCASTIC_K'] - data['STOCASTIC_D']
    data['B_delta'] = data.Bollinger.diff().fillna(0)
    data['MACD_delta'] = data.MACD.diff().fillna(0)
#     ma20 = new_gs['Adj Close'].rolling(window=20).mean()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA5_adj'] = (data['MA5'] - data['Close']) / data['Close']
    data['MA20_adj'] = (data['MA20'] - data['Close']) / data['Close']
    data['MA_diff'] = (data['MA5'] - data['MA20']) / data['Close']

def chart_today_predict(last_day):
    cat_model = settings.CAT_MODEL

    mean = pd.read_csv('./data/mean.csv', index_col = 0).drop('Change+', axis = 0)
    std = pd.read_csv('./data/std_df.csv', index_col = 0).drop('Change+', axis = 0)

    data = fdr.DataReader('069500', '2022', last_day)
    all(data)

    data.replace([np.inf, -np.inf], np.nan, inplace = True)
    data.dropna(inplace = True)
    data_1 = data.iloc[[-1]][['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD',  'MACD_SIGNAL', 'MACD_cat', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
                       'MA5', 'MA20', 'MA5_adj', 'MA20_adj', 'MA_diff']]

    data = data.iloc[-1][['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD',  'MACD_SIGNAL', 'MACD_cat', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
                       'MA5', 'MA20', 'MA5_adj', 'MA20_adj', 'MA_diff']]

    data_1.reset_index(drop = True, inplace = True)

    data = (np.array(data) - np.array(mean['mean'])) / np.array(std['std'])

    data_1.loc[0] = data  # 오늘자 표준화 완료된 데이터프레임

    data_1['MACD_cat'] = 0

    data_1.insert(6, 'MACD_cat2', 0)

    data_1.drop('MA_diff', axis = 1, inplace = True)

    prediction = cat_model.predict_proba(np.array(data_1))
    up = prediction[0][0]
    down = prediction[0][1]
    return up, down
