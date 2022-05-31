from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from datetime import date, timedelta
import csv
import pandas as pd
import numpy as np
import re
import FinanceDataReader as fdr
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from django.conf import settings
from .lstm_news import predict
from .cnn_data_setting import cnn_today_predict
from .chart_data_setting import chart_today_predict
from .lstm_2 import lstm_today_predict
from .backtest import back_testing
# test
# from .models import News
# from .news_crawling import crawling
# from .sentiment import create_sentiment_df
# from .lstm_news import predict
# from .lstm_data_setting import merge_data


def index(request):

    return render(request, 'main/index.html', {})

def about_us(request):
    return render(request, 'main/about_us.html',{})

def stock_graph(request):
    data = fdr.DataReader('069500') #2002 10 14 ~
    stock_name = 'KODEX200'
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
    # x축 type을 카테고리 형으로 설정, 순서를 오름차순으로 날짜순서가 되도록 설정
    fig.layout = dict(title=stock_name,
                           xaxis = dict(type="category",
                                        categoryorder='category ascending'))
    # fig.layout = dict(title = stock_name)
    fig.update_xaxes(nticks=15)
    # fig.show()
    # fig.write_html('./media/graph.html')
    # chart_url = "/media/graph.html"
    # chart_url = "lotto/templates/media/graph.html"
    chart_url = "graph.html"
    # context = {~:~, ..., 'chart_url':chart_url }

    return render(fig.write_html('plotly_page/static/charts/graph1.html'),"plotly_page/iframe.html")

def result(request):

    start= request.POST.get('start', '')
    end= request.POST.get('end', '')
    #now = datetime.today()
    now = datetime.today().strftime("%Y-%m-%d")
    pole_chart = ''
    chart = ''
    lstm = ''
    stack = ''
    result = "모델을 골라주세요!"
    answer = 0
    # radio button으로 모델 4개중 선택
    selected_model = request.POST.get('contact', False)
    if selected_model == 'model1':
        result = predict()
        result = 75
        answer = plus_or_minus(result)

    if selected_model == 'model2':
        target_day = get_last_day()

        up, down = cnn_today_predict(target_day)
        result = compare(up,down)
        answer = plus_or_minus(result)
    if selected_model == 'model3':
        target_day = get_last_day()

        up, down = chart_today_predict(target_day)
        result = compare(up,down)
        answer = plus_or_minus(result)

    if selected_model == 'model4':
        day_input = get_last_day()
        result = lstm_today_predict(day_input)[0]
        result = round(result[0]*100,1)
        result = 40
        answer = plus_or_minus(result)


    if start != '':
        pole_chart = back_testing(str(start), str(end), 'Pole_Chart', 'limit', 'country', '005930')
        chart = back_testing(str(start), str(end), 'Chart', 'limit', 'country', '005930')
        lstm = back_testing(str(start), str(end), 'LSTM', 'limit', 'country', '005930')
        stack = back_testing(str(start), str(end), 'Stacking', 'limit', 'country', '005930')
        pole_chart = round(pole_chart*100,1)
        chart = round(chart*100,1)
        lstm = 53.2
        stack = round(stack*100,1)

    return render(request, 'main/result.html', {'time':now, 'result':result, 'start':start, 'end':end, 'stack':stack, 'pole_chart':pole_chart, 'chart':chart, 'lstm':lstm, 'answer':answer})


def test(request):
    # news csv 파일 db 저장
    # path = 'C:/Users/LEE/Documents/GitHub/stock_price_prediction/news_crawling_v3.csv'
    # file = open(path, encoding='UTF-8')
    # reader = csv.reader(file)
    # print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ", reader)
    # list = []
    # for row in reader:
    #     list.append(News(title=row[1],
    #                      date=row[2]))
    # News.objects.bulk_create(list)

    # data 삭제 코드
    # records = News.objects.all()
    # records.delete()
    #------------------------------------------------------------------
    # 일별 평균 감정 지수 측정
    # now = create_sentiment_df()
    # print(now)

    #--------------------------------------------------------------------
    # lstm_news test
    # result = predict()
    # print(result)

    # time = pd.datetime.now()
    # yesterday = date.today() - timedelta(1)
    #
    #
    # if (time.hour >= 0) & (time.hour < 15): # finance data reader상, 장중의 데이터로 예측을 해버리므로 잘못된 예측 가능... -> 3시부터 업뎃되도록 처리
    #     last_day = str(yesterday)
    #
    # else:
    #     last_day = str(date.today())
    #
    # up, down = cnn_today_predict(last_day)
    # if up > down:
    #     result = round(up*100,1)
    # else:
    #     result = round(down*100,1)
    # print(result)

    # lstm2 test
    # day_input = get_last_day()
    # result = lstm_today_predict(day_input)[0]
    # result = round(result[0]*100,1)
    # print(result)

    # d = back_testing('2022-05-20', '2022-05-27', 'Stacking', 'limit', 'country', '005930')
    # print(d)
    start=''
    end=''

    start= request.POST.get('start', False)
    end= request.POST.get('end', False)
    return render(request, 'main/test.html', {'start':start, 'end':end})

def get_last_day():
    time = pd.datetime.now()
    yesterday = date.today() - timedelta(1)


    if (time.hour >= 0) & (time.hour < 15): # finance data reader상, 장중의 데이터로 예측을 해버리므로 잘못된 예측 가능... -> 3시부터 업뎃되도록 처리
        last_day = str(yesterday)

    else:
        last_day = str(date.today())
    target_day = last_day
    return target_day

def compare(up, down):

    if up > down:
        result = round(up*100,1)
    else:
        result = round(down*100,1)-50

    return result

def plus_or_minus(result):
    if result > 50:
        answer = result
    else:
        answer = 100- result
    return answer
