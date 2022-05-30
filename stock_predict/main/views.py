from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from datetime import date, timedelta
import csv
import pandas as pd
import numpy as np
import re

from django.conf import settings
from .lstm_news import predict
from .cnn_data_setting import cnn_today_predict
from .chart_data_setting import chart_today_predict
from .lstm_2 import lstm_today_predict
# test
# from .models import News
# from .news_crawling import crawling
# from .sentiment import create_sentiment_df
# from .lstm_news import predict
# from .lstm_data_setting import merge_data


def index(request):

    return render(request, 'main/index.html', {})

def result(request):

    #now = datetime.today()
    now = datetime.today().strftime("%Y-%m-%d")
    result = "모델을 골라주세요!"
    # radio button으로 모델 4개중 선택
    selected_model = request.POST.get('contact', False)
    if selected_model == 'model1':
        result = predict()

    if selected_model == 'model2':
        target_day = get_last_day()

        up, down = cnn_today_predict(target_day)
        result = compare(up,down)

    if selected_model == 'model3':
        target_day = get_last_day()

        up, down = chart_today_predict(target_day)
        result = compare(up,down)

    if selected_model == 'model4':
        day_input = get_last_day()
        result = lstm_today_predict(day_input)[0]
        result = round(result[0]*100,1)

    return render(request, 'main/result.html', {'time':now, 'result':result})


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
    day_input = get_last_day()
    result = lstm_today_predict(day_input)[0]
    result = round(result[0]*100,1)
    print(result)
    return HttpResponse("asddsdasf")

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
