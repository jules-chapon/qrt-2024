"""Pipeline of the project"""

from predict_foot_result.configs import constants

from predict_foot_result.libs.preprocessing import preprocessing_learning

from predict_foot_result.model.lgbm_model import LgbmClassificationModel

from predict_foot_result.model.dummy_model import DummyClassificationModel


def learning_pipeline_lgbm():
    df_learning = preprocessing_learning()
    model = LgbmClassificationModel("model_lgbm_1")
    model.training_pipeline(df_learning)
    return None


def learning_pipeline_dummy():
    df_learning = preprocessing_learning()
    model = DummyClassificationModel("model_dummy_1")
    model.training_pipeline(df_learning)
    return None
