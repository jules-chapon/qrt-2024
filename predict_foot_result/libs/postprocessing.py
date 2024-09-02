"""Postprocessing functions"""

import os
import numpy as np
import pandas as pd
from logger import logging

from predict_foot_result.configs import names


def add_predictions_to_dataset(
    df_testing: pd.DataFrame, array_predictions: np.ndarray
) -> pd.DataFrame:
    """
    Add predictions to the testing dataset.

    Args:
        df_testing (pd.DataFrame): Dataframe with testing data.
        array_predictions (np.ndarray): Array containing predictions.

    Returns:
        pd.DataFrame: Dataframe with predictions and IDs.
    """
    df_testing[names.PREDICTED_LABEL] = array_predictions
    df_predictions = df_testing[[names.ID, names.PREDICTED_LABEL]].copy()
    return df_predictions


def clean_predictions_format(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the predictions table to match the expected format.

    Args:
        df_predictions (pd.DataFrame): DataFrame with predictions.

    Returns:
        pd.DataFrame: DataFrame with cleaned predictions.
    """
    df_predictions[names.HOME_WINS] = np.where(
        df_predictions[names.PREDICTED_LABEL] == 0,
        1,
        0,
    )
    df_predictions[names.DRAW] = np.where(
        df_predictions[names.PREDICTED_LABEL] == 1,
        1,
        0,
    )
    df_predictions[names.AWAY_WINS] = np.where(
        df_predictions[names.PREDICTED_LABEL] == 2,
        1,
        0,
    )
    df_predictions.drop(columns=[names.PREDICTED_LABEL], inplace=True)
    df_predictions.columns = [col.upper() for col in df_predictions.columns]
    logging.info("Predictions have been formatted")
    return df_predictions


def save_predictions(df_predictions: pd.DataFrame, file_name: str) -> None:
    """
    Save predictions to a CSV file.

    Args:
        df_predictions (pd.DataFrame): DataFrame with predictions.
        file_name (str): Name of the CSV file to save predictions to.
    """
    df_predictions.to_csv(
        os.path.join(names.DATA_FOLDER, names.RESULT_FOLDER, file_name), index=False
    )
    logging.info(f"Saving { df_predictions.shape[0] } predictions")
    logging.info(f"Path : { os.path.join(names.DATA_FOLDER, names.RESULT_FOLDER, file_name) }")


def postprocessing_pipeline(
    df_testing: pd.DataFrame, array_predictions: np.ndarray, file_name: str
) -> None:
    """
    Postprocessing pipeline for predictions.

    Args:
        df_testing (pd.DataFrame): DataFrame with test data.
        array_predictions (np.ndarray): Array containing predictions.
        file_name (str): Name of the CSV file to save predictions to.
    """
    df_predictions = add_predictions_to_dataset(df_testing, array_predictions)
    df_predictions = clean_predictions_format(df_predictions)
    save_predictions(df_predictions, file_name)
