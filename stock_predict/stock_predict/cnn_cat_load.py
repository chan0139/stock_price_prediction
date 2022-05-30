from django.conf import settings
import keras
from catboost import CatBoostClassifier, Pool
def cnn_cat_load():

    base_url = settings.MODEL_ROOT_URL + settings.MODEL_URL # == './model/'
    model_url = base_url + 'cnn_final_model'
    cnn_model = keras.models.load_model('./model/cnn_final_model')

    cat_model = CatBoostClassifier()
    cat_model.load_model("./model/Cat_model")

    return cnn_model, cat_model
