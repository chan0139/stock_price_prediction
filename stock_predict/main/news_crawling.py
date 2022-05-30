import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from datetime import datetime


import pandas as pd


def crawling():
    # target_url 저장 후 업뎃시킬 방안 생각해야됨.
    target_url = 'https://finance.naver.com/news/mainnews.naver?date=' + '2022-05-20'
    article_df = pd.DataFrame()

    yesterday = datetime.today() - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")

    url = 'https://finance.naver.com/news/mainnews.naver?date=' + yesterday
    while url != target_url:
        res = requests.get(url).content
        soup = BeautifulSoup(res, 'html.parser')

        #headline 추출
        news_subjects = soup.find_all('dd', {'class' : 'articleSubject'})

        subject_list = []

        for subject in news_subjects:
            subject_list.append(subject.get_text().strip())
        #날짜 추출
        date = soup.find('span', {'class' : 'viewday'}).get_text()
        date = url[51:55] + '년' + date

        if date[-2] != '토' and date[-2] != '일':
            temp = pd.DataFrame({'Title':subject_list,
                                       'Date':date, })
            article_df = pd.concat([article_df, temp])


        date_list = soup.find('div', {'class' : 'pagenavi_day'})
        rest_url = date_list.find_all('a')[2]['href']

        # 다음 크롤링할 target url
        url = 'https://finance.naver.com' + rest_url


    return article_df
