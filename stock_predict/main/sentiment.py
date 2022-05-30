import re
from .news_crawling import crawling
from django.conf import settings
import numpy as np
import pandas as pd

# tokenizer = settings.MY_TOKENIZER
model = settings.MODEL_KOBERT

def delete_bracket(row):
    x = row['Title']
    pattern = r'\[([^]]+)\]'
    #x = '이건 [괄호 안의 불필요한 정보를] 삭제하는 코드다.' test code

    text = re.sub(pattern=pattern, repl='', string= x)
    return text

def encoding(row):
    sentence = row['new'] # title encoding -> new
    SEQ_LEN = 64 # 최대 token 개수 이상의 값으로 임의로 설정

    # Tokenizing / Tokens to sequence numbers / Padding
    encoded_dict = settings.MY_TOKENIZER.encode_plus(text=re.sub("[^\s0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]", "", sentence),
                                         padding='max_length',
                                         truncation = True,
                                         max_length=SEQ_LEN) # SEQ_LEN == 128

    token_ids = np.array(encoded_dict['input_ids']).reshape(1, -1) # shape == (1, 128) : like appending to a list
    token_masks = np.array(encoded_dict['attention_mask']).reshape(1, -1)
    token_segments = np.array(encoded_dict['token_type_ids']).reshape(1, -1)
    new_inputs = (token_ids, token_masks, token_segments)
    return new_inputs

def predict_sentiment(row):
    encode_sentence = row['encode']
    # Prediction
    prediction = model.predict(encode_sentence)
    predicted_probability = np.round(np.max(prediction) * 100, 2) # ex) [[0.0125517 0.9874483]] -> round(0.9874483 * 100, 2) -> round(98.74483, 2) -> 98.74
    predicted_class = ['부정', '긍정'][np.argmax(prediction, axis=1)[0]] # ex) ['부정', '긍정'][[1][0]] -> ['부정', '긍정'][1] -> '긍정'


    #print("{}% 확률로 {} 리뷰입니다.".format(predicted_probability, predicted_class))
    return predicted_class, predicted_probability

def grouping_date(df):
    df.loc[df['percent'] < 70, 'label'] = '중립'
    df.loc[(df['label'] == '부정') & (df['percent'] < 80), 'label'] ='중립'
    df.loc[df['label'] == '긍정', 'label_index'] = 1
    df.loc[df['label'] == '중립', 'label_index'] = 0
    df.loc[df['label'] == '부정', 'label_index'] = -1

    groups = df.groupby('Date')
    date_sentiment = groups.mean()
    return date_sentiment

def create_sentiment_df():
    article_df = crawling()

    article_df['new'] = article_df.apply(delete_bracket, axis='columns')
    article_df['encode'] = article_df.apply(encoding, axis='columns')
    article_df['label'], article_df['percent']  = zip(*article_df.apply(predict_sentiment, axis='columns'))

    date_sentiment = grouping_date(article_df)
    return date_sentiment
