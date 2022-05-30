import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import cv2
import numpy as np
from django.conf import settings

def draw_picture_today(last_day):     #예측 후 삭제 과정 필요 # 20일로 일단 디폴트

    local_data = fdr.DataReader('069500', '2022', last_day)[-20:]


    up = local_data[local_data.Close >= local_data.Open]
    down = local_data[local_data.Close < local_data.Open]

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
    plt.savefig('./picture/picture.png')

#     image = np.array([cv2.imread('picture/picture.png')])
    # image = np.reshape(image, (1, image.shape[0], image.shape[1]))

def cnn_today_predict(last_day):

    cnn_model = settings.CNN_MODEL

    draw_picture_today(last_day)

    image = np.array([cv2.imread('./picture/picture.png')])

    prediction = cnn_model.predict(image)
    up = prediction[0][0]
    down = prediction[0][1]



    return up,down
