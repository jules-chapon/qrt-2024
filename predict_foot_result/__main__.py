"""Pipeline of the project"""

from predict_foot_result.libs.preprocessing import (
    preprocessing_learning,
    preprocessing_testing,
)

from predict_foot_result.libs.postprocessing import postprocessing_pipeline

from predict_foot_result.model.lgbm_model import LgbmClassificationModel


def learning_pipeline_lgbm(model_name: str) -> None:
    df_learning = preprocessing_learning()
    model = LgbmClassificationModel(model_name)
    model.training_pipeline(
        df_learning=df_learning,
        is_balanced_data=True,
        feature_selection=False,
        fine_tuning=True,
    )
    model.save_model()
    return None


def prediction_pipeline_lgbm(model_name: str) -> None:
    df_testing = preprocessing_testing()
    model = LgbmClassificationModel.load_model(model_name)
    array_predictions = model.predict(df_testing)
    postprocessing_pipeline(
        df_testing, array_predictions, f"{ model_name }_predictions.csv"
    )
    return None


if __name__ == "__main__":
    model_name = "model_test"
    learning_pipeline_lgbm(model_name)
    prediction_pipeline_lgbm(model_name)
