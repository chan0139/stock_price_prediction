from django.conf import settings
import keras

def lstm_load():

    base_url = settings.MODEL_ROOT_URL + settings.MODEL_URL # == './model/'
    model_url = base_url + 'lstm_news_model.h5'
    model = keras.models.load_model(model_url)

    return model
