from django.conf import settings
import numpy as np
import FinanceDataReader as fdr

def lstm_today_predict(last_day):
    model = settings.LSTM_MODEL_2

    local_data = fdr.DataReader("069500", '2020', last_day)[-60:]

    local_data = np.array(local_data)

    local_data = np.reshape(local_data, (1, 60, 6))

    prediction = model.predict(local_data)

    return prediction
