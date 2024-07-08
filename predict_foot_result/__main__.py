"""Pipeline of the project"""

from predict_foot_result.configs import constants

from predict_foot_result.libs.preprocessing import preprocessing_learning

from predict_foot_result.model.lgbm_model import LgbmClassificationModel


def learning_pipeline():
    df_learning = preprocessing_learning()
    model = LgbmClassificationModel("model_1")
    model.training_pipeline(df_learning)
    return None
