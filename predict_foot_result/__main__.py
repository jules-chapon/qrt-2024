"""Pipeline of the project"""

from predict_foot_result.libs.preprocessing import (
    preprocessing_learning,
    preprocessing_testing,
)

from predict_foot_result.libs.postprocessing import postprocessing_pipeline

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


def prediction_pipeline_lgbm(model_name: str) -> None:
    df_testing = preprocessing_testing()
    model = LgbmClassificationModel.load_model(model_name)
    array_predictions = model.predict(df_testing)
    postprocessing_pipeline(
        df_testing, array_predictions, f"{ model_name }_predictions.csv"
    )
    return None
