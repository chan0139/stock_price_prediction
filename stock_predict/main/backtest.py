import numpy as np
import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from django.conf import settings
from sklearn.metrics import accuracy_score

def draw_picture_days(code, start_day, end_day):     #예측 후 삭제 과정 필요 # 20일로 일단 디폴트

    cnn_model = settings.CNN_MODEL

    month_before = (datetime.strptime(start_day, '%Y-%m-%d') - relativedelta(months = 2)).strftime('%Y-%m-%d')  # str

    local_data = fdr.DataReader(code, month_before, end_day)

    start_number = local_data.index.strftime("%Y-%m-%d").tolist().index(start_day)

    end_number = len(local_data)

    x = 0
    y = 0

    for i in range(start_number, end_number):

        data = local_data.iloc[i - 20 : i]
        change_1 = local_data.iloc[i]['Change']
        change = (local_data.iloc[i]['Change'] > 0).astype(int)

        up = data[data.Close >= data.Open]
        down = data[data.Close < data.Open]

        width = 1
        width2 = .1

        col1 = 'red'
        col2 = 'blue'

        plt.cla()
        plt.style.use('dark_background')
        plt.figure(figsize=(1,1), dpi = 50)

        plt.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
        plt.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
        plt.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)

        plt.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
        plt.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
        plt.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)

        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        plt.axis('off')

    #         plt.savefig('drive/My Drive/datasets/datasets/{}_{}_{}.png'.format(variables, i, label), dpi = 1000)
        plt.savefig('./picture/picture2.png')

        image = np.array([cv2.imread('./picture/picture2.png')])   # 50 50 3

        null_array = np.array([[0, 0]])

        if change > 0:
            null_array[0][1] = 1
        else:
            null_array[0][0] = 1

        result = cnn_model.evaluate(image, null_array)

        x += result[1]
        y += 1

    return x / y

def draw_picture_days2(code, start_day, end_day):     #예측 후 삭제 과정 필요 # 20일로 일단 디폴트

    cnn_model = settings.CNN_MODEL

    month_before = (datetime.strptime(start_day, '%Y-%m-%d') - relativedelta(months = 2)).strftime('%Y-%m-%d')  # str

    local_data = fdr.DataReader(code, month_before, end_day)

    start_number = local_data.index.strftime("%Y-%m-%d").tolist().index(start_day)

    end_number = len(local_data)

    x = []
    y = []

    for i in range(start_number, end_number):

        data = local_data.iloc[i - 20 : i]
        change_1 = local_data.iloc[i]['Change']
        change = (local_data.iloc[i]['Change'] > 0).astype(int)

        up = data[data.Close >= data.Open]
        down = data[data.Close < data.Open]

        width = 1
        width2 = .1

        col1 = 'red'
        col2 = 'blue'

        plt.cla()
        plt.style.use('dark_background')
        plt.figure(figsize=(1,1), dpi = 50)

        plt.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
        plt.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
        plt.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)

        plt.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
        plt.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
        plt.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)

        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        plt.axis('off')

    #         plt.savefig('drive/My Drive/datasets/datasets/{}_{}_{}.png'.format(variables, i, label), dpi = 1000)
        plt.savefig('./picture/picture2.png')

        image = np.array([cv2.imread('./picture/picture2.png')])   # 50 50 3

        null_array = np.array([[0, 0]])

        if change > 0:
            null_array[0][1] = 1
        else:
            null_array[0][0] = 1

        result = cnn_model.predict(image)

        x.append(result)

    return x



def draw_stacking_accuracy(code, start_day, end_day):
    a, y = draw_chart_accuracy2(code, start_day, end_day)
    b = draw_picture_days2(code, start_day, end_day)

    x = []
    for i, j in zip(a, b):
        x.append(((i[0] + j[0][0]) / 2, (i[1] + j[0][1]) / 2))
    for i in range(len(x)):
        if x[i][0] > x[i][1]:
            x[i] = 0
        else:
            x[i] = 1
    return accuracy_score(x, list(y))
#     return accuracy_score(x, y)
def draw_chart_accuracy(code, day_start, end_day):

    try:
        month_before = (datetime.strptime(day_start, '%Y-%m-%d') - relativedelta(months = 2)).strftime('%Y-%m-%d')  # str

        data = fdr.DataReader(str(code), month_before, end_day)

        all(data)

        data.replace([np.inf, -np.inf], np.nan, inplace = True)

        data.dropna(inplace = True)

        data = data.loc[day_start :]            # 일종의 문법

        y = data['Change+']

        data = data[['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD', 'MACD_cat', 'MACD_SIGNAL', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
                           'MA5', 'MA20', 'MA5_adj', 'MA20_adj']]

        data.reset_index(drop = True, inplace = True)

        mean = pd.read_csv('./data/mean.csv', index_col = 0).drop('Change+', axis = 0).T
        mean2 = mean.copy()
        std = pd.read_csv('./data/std_df.csv', index_col = 0).drop('Change+', axis = 0).T
        std2 = std.copy()


        while len(mean) != len(data):
            mean = pd.concat([mean, mean2])
            std = pd.concat([std, std2])

        mean.reset_index(drop = True, inplace = True)

        std.reset_index(drop = True, inplace = True)

        mean = mean[['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD',
        'MACD_cat', 'MACD_SIGNAL', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
        'MA5', 'MA20', 'MA5_adj', 'MA20_adj', 'MA_diff']]

        std = std[['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD',
        'MACD_cat', 'MACD_SIGNAL', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
        'MA5', 'MA20', 'MA5_adj', 'MA20_adj', 'MA_diff']]

        mean.drop('MA_diff', axis = 1, inplace = True)

        std.drop('MA_diff', axis = 1, inplace = True)

        data = (data - mean) / std

        data.drop('MACD_cat', axis = 1, inplace = True)

        data['MACD_cat'] = 1

        data['MACD_cat2'] = 0
        cat_model = settings.CAT_MODEL
        prediction = cat_model.predict(np.array(data))

        accuracy = accuracy_score(y, prediction)

        return accuracy

    except:
        return 0.5

def draw_chart_accuracy2(code, start_day, end_day):

    month_before = (datetime.strptime(start_day, '%Y-%m-%d') - relativedelta(months = 2)).strftime('%Y-%m-%d')  # str

    data = fdr.DataReader(str(code), month_before, end_day)

    all(data)

    data.replace([np.inf, -np.inf], np.nan, inplace = True)

    data.dropna(inplace = True)

    data = data.loc[start_day :]            # 일종의 문법

    y = data['Change+']

    data = data[['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD', 'MACD_cat', 'MACD_SIGNAL', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
                       'MA5', 'MA20', 'MA5_adj', 'MA20_adj']]

    data.reset_index(drop = True, inplace = True)

    mean = pd.read_csv('./data/mean.csv', index_col = 0).drop('Change+', axis = 0).T
    mean2 = mean.copy()
    std = pd.read_csv('./data/std_df.csv', index_col = 0).drop('Change+', axis = 0).T
    std2 = std.copy()


    while len(mean) != len(data):
        mean = pd.concat([mean, mean2])
        std = pd.concat([std, std2])

    mean.reset_index(drop = True, inplace = True)

    std.reset_index(drop = True, inplace = True)

    mean = mean[['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD',
    'MACD_cat', 'MACD_SIGNAL', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
    'MA5', 'MA20', 'MA5_adj', 'MA20_adj', 'MA_diff']]

    std = std[['RSI', 'STOCASTIC_K', 'STOCASTIC_D', 'Bollinger', 'MACD',
    'MACD_cat', 'MACD_SIGNAL', 'RSI_delta', 'D_delta', 'sto_diff', 'B_delta', 'MACD_delta',
    'MA5', 'MA20', 'MA5_adj', 'MA20_adj', 'MA_diff']]

    mean.drop('MA_diff', axis = 1, inplace = True)

    std.drop('MA_diff', axis = 1, inplace = True)

    data = (data - mean) / std

    data.drop('MACD_cat', axis = 1, inplace = True)

    data['MACD_cat'] = 1

    data['MACD_cat2'] = 0
    cat_model = settings.CAT_MODEL
    prediction = cat_model.predict_proba(np.array(data))

    return prediction, y

def back_testing(start_day, end_day, model, limit, country, code):  # ('2020-12-01', '2021-3-1', 'CNN', )

    if model == 'Pole_Chart':

        cnn_accuracy = draw_picture_days(code, start_day, end_day)

        return cnn_accuracy

    if model == 'Chart':

        chart_accuracy = draw_chart_accuracy(code, start_day, end_day)

        return chart_accuracy



    if model == 'Stacking':

        Voting_accuracy = draw_stacking_accuracy(code, start_day, end_day)

        return Voting_accuracy

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
