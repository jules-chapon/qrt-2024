"""Class for dummy model"""

import numpy as np
import pandas as pd

from predict_foot_result.configs import names, constants
from predict_foot_result.model.classification_model import (
    ClassificationModel,
    _ClassificationModel,
)


class DummyClassificationModel(ClassificationModel):
    """
    Class to represent dummy classification models.
    We always predict home team wins.

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(
        self: _ClassificationModel,
        name: str,
        path: str | None = names.MODELS_FOLDER,
        target: str | None = constants.TARGET,
        features: list | str | None = None,
        params: dict | None = None,
        cols_id: list | str | None = names.ID,
        train_valid_split: float = constants.TRAIN_VALID_SPLIT,
        metrics: dict | None = None,
    ) -> None:
        """
        Initialize class object.

        Args:
            self (_ClassificationModel): Class object.
            name (str): Name of the model.
            path (str | None, optional): Path to the model. Defaults to names.MODELS_FOLDER.
            target (str | None, optional): Target of the model. Defaults to constants.TARGET.
            features (list | str | None, optional): Features of the model. Defaults to None.
            params (dict | None, optional): Hyperparameters of the model. Defaults to None.
            cols_id (list | str | None, optional): Columns used as IDs. Defaults to None.
            train_valid_split (float, optional): Train-validation split.
                Defaults to constants.TRAIN_VALID_SPLIT.
            metrics (dict | None, optional): Metrics to evaluate the model. Defaults to None.
        """
        super().__init__(
            name, path, target, features, params, cols_id, train_valid_split, metrics
        )

    def train(
        self: _ClassificationModel,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        """
        Train the model and store it to self.model.

        Args:
            self (_ClassificationModel): Class object.
            df_train (pd.DataFrame): Training set.
            df_valid (pd.DataFrame): Validation set.
        """

    def predict(
        self: _ClassificationModel,
        df_to_predict: pd.DataFrame,
    ) -> pd.Series:
        """
        Get predictions of the model.

        Args:
            self (_ClassificationModel): Class object.
            df_to_predict (pd.DataFrame): DataFrame to make predictions with.

        Returns:
            pd.Series: Predicted values.
        """
        predictions_label = np.full(df_to_predict.shape[0], 0)
        return predictions_label

    def fine_tune(
        self: _ClassificationModel,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        """
        Fine-tune hypermarameters of the model.

        Args:
            df_train (pd.DataFrame): Training set.
            df_valid (pd.DataFrame): Validation set.
        """
