import warnings
import FinanceDataReader as fdr
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from .sentiment import create_sentiment_df
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Conv1D

warnings.filterwarnings('ignore')

def get_sentiment():
    added_sentiment = create_sentiment_df()
    added_sentiment = added_sentiment.drop(['percent'], axis=1)
    added_sentiment = added_sentiment.reset_index()
    # read sentiment data & cleaning
    date_sentiment = pd.read_csv('./data/date_sentiment.csv')
    date_sentiment = date_sentiment.drop(['Unnamed: 0', 'Unnamed: 0.1', 'percent'], axis=1)
    now_date_sentiment = pd.concat([date_sentiment, added_sentiment])

    now_date_sentiment = now_date_sentiment.reset_index()
    #
    for i in range(len(now_date_sentiment)):
        now_date_sentiment['Date'][i] = now_date_sentiment['Date'][i][:-4]

    for i in range(len(now_date_sentiment)):
        now_date_sentiment['Date'][i] = now_date_sentiment['Date'][i].replace('년','-').replace('월','-')

    now_date_sentiment['Date'] = pd.to_datetime(now_date_sentiment['Date'])
    now_date_sentiment = now_date_sentiment.set_index('Date')

    return now_date_sentiment

def get_stock_data():
    data = fdr.DataReader('069500',start='2007-01-03')
    data = data.reset_index()

    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']
    data['Date'] = pd.to_datetime(data['Date'])

    data.index = data['Date']
    data.set_index('Date', inplace=True)

    target = []
    for i in range(data.shape[0]):
        if data['Change'][i] >0:
            target.append(1)
        else:
            target.append(0)
    data['Target'] = target
    return data

def get_economy_data():
    nasdaq_df = fdr.DataReader('NASDAQCOM', data_source='fred', start = '2007-01-02')
    dji_df = fdr.DataReader('DJI', '2007-01-01')
    gold_df = fdr.DataReader('ZG', start='2007-01-01')
    usdkrw_df = fdr.DataReader('USD/KRW', '2007-01-01') # 달러 원화
    usdeur_df = fdr.DataReader('USD/EUR', '2007-01-01') # 달러 유로화
    usdcny_df = fdr.DataReader('USD/CNY', '2007-01-01') # 달러 위엔화

    return nasdaq_df, dji_df,gold_df,usdkrw_df,usdeur_df,usdcny_df

# merge sentiment & economy data
def make_feature_data():
    sentiment_df = get_sentiment()
    date_sentiment = sentiment_df['label_index']
    nasdaq_df, dji_df,gold_df,usdkrw_df,usdeur_df,usdcny_df = get_economy_data()
    nasdaq = nasdaq_df['NASDAQCOM']
    dji = dji_df['Close']
    gold = gold_df['Close']
    usdkrw = usdkrw_df['Close']
    usdeur = usdeur_df['Close']
    usdcny = usdcny_df['Close']

    merge_df = pd.concat([nasdaq, dji, gold, usdkrw, usdeur, usdcny,date_sentiment],axis=1, join='inner')   #열방향(axis=1), 교집합(inner)
    columns=['NASDAQ','DJI', 'GOLD', 'KRW', 'EUR', 'CNY', 'sentiment']
    merge_df.columns= columns
    return merge_df

# kodex, merge_df 중복된 일자만 남기기 (누락일자 제외 작업)
def merge_data():
    data = get_stock_data()
    merge_df = make_feature_data()
    data_merge_df = pd.concat([data, merge_df],axis=1, join='inner')
    return data_merge_df

def scaling():
    scaler = StandardScaler()

    data_merge_df = merge_data()
    kodex_df = data_merge_df.iloc[:,:7]
    economy_df = data_merge_df.iloc[:,7:]
    label= kodex_df['Target']

    columns=['NASDAQ','DJI', 'GOLD', 'KRW', 'EUR', 'CNY', 'sentiment']
    scaled_data = scaler.fit_transform(economy_df[columns])
    scaled_economy_df = pd.DataFrame(scaled_data, columns=columns)


    columns = ['Open', 'High', 'Low', 'Close', 'Volume','Change', 'Target']
    scaled_data = scaler.fit_transform(kodex_df[columns])
    scaled_kodex_df = pd.DataFrame(scaled_data, columns=columns)

    volume = scaled_kodex_df['Volume']
    scaled_economy_df = pd.concat([scaled_economy_df,volume],axis=1)

    return scaled_economy_df, scaled_kodex_df, label
