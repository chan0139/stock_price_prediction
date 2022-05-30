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
from .cnn_data_setting import draw_picture_today

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
        result = "CNN"

    if selected_model == 'model3':
        result = "LSTM"

    if selected_model == 'model4':
        result = "CHART"
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

    time = pd.datetime.now()
    yesterday = date.today() - timedelta(1)


    if (time.hour >= 0) & (time.hour < 15): # finance data reader상, 장중의 데이터로 예측을 해버리므로 잘못된 예측 가능... -> 3시부터 업뎃되도록 처리
        last_day = str(yesterday)

    else:
        last_day = str(date.today())

    prediction = draw_picture_today(last_day)
    


    return HttpResponse("asddsdasf")
