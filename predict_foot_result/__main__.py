"""Pipeline of the project"""

from predict_foot_result.libs.preprocessing import (
    preprocessing_learning,
    preprocessing_testing,
)

from predict_foot_result.libs.postprocessing import postprocessing_pipeline

from predict_foot_result.model.lgbm_model import LgbmClassificationModel


def learning_pipeline_lgbm(model_name: str) -> None:
    """
    This function is responsible for executing the learning pipeline using the LightGBM model.

    Args:
        model_name (str): The name of the model to be used for training.
    """
    df_learning = preprocessing_learning()
    model = LgbmClassificationModel(model_name)
    model.training_pipeline(
        df_learning=df_learning,
        is_balanced_data=True,
        feature_selection=False,
        fine_tuning=True,
    )
    model.save_model()


def prediction_pipeline_lgbm(model_name: str) -> None:
    """
    This function executes the prediction pipeline using the LightGBM model.

    Parameters:
        model_name (str): The name of the model to be used for prediction.
            This name should match the name used when saving the model.
    """
    df_testing = preprocessing_testing()
    model = LgbmClassificationModel.load_model(model_name)
    array_predictions = model.predict(df_testing)
    postprocessing_pipeline(df_testing, array_predictions, f"{ model_name }_predictions.csv")


if __name__ == "__main__":
    name = "test"
    learning_pipeline_lgbm(name)
    prediction_pipeline_lgbm(name)
