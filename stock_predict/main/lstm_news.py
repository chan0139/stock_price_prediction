from .lstm_data_setting import scaling
from django.conf import settings
import numpy as np

def predict():
    model = settings.LSTM_MODEL

    scaled_economy_df, scaled_kodex_df, label = scaling()

    window_size = 10
    sequence_length = window_size + 1

    result = []
    label_list = []
    for index in range(len(scaled_economy_df) - sequence_length):
        result.append(scaled_economy_df[index:index+window_size])
        label_list.append(label[index+window_size])

    x = np.array(result)
    y = np.array(label_list)

    split = -100

    x_test = x[split:]
    y_test = y[split:]

    temp = np.reshape(x_test[-1], (1, x_test[-1].shape[0],x_test[-1].shape[1]))

    result = model.predict(temp)[0]
    result = round(result[0]*100,1)

    return result
