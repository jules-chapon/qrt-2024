"""Class for LGBM model"""

import os
from typing import List
import lightgbm as lgb
from logger import logging
import pickle as pkl
import numpy as np
import pandas as pd

from predict_foot_result.configs import constants
from predict_foot_result.model.classification_model import (
    ClassificationModel,
    _ClassificationModel,
)


class LgbmClassificationModel(ClassificationModel):
    """
    Class to represent LGBM classification models.

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(
        self: _ClassificationModel,
        name: str,
        path: str | None = None,
        target: str | None = constants.TARGET,
        features: list | str | None = None,
        params: dict | None = constants.LGBM_PARAMS,
        cols_id: list | str | None = constants.LGBM_ID,
        train_valid_split: float = constants.TRAIN_VALID_SPLIT,
        metrics: dict | None = None,
    ) -> None:
        """
        Initialize class object.

        Args:
            self (_ClassificationModel): Class object.
            name (str): Name of the model.
            path (str | None, optional): Path to the model. Defaults to None.
            target (str | None, optional): Target of the model. Defaults to None.
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

    def define_features(
        self: _ClassificationModel,
        df_learning: pd.DataFrame,
        list_features: List[str] | None = None,
        list_cols_to_drop: List[str] | None = constants.LGBM_ID
        + [constants.LGBM_LABEL],
    ) -> None:
        """
        Define features for the model.

        Args:
            self (_ClassificationModel): Class object.
            df_learning (pd.DataFrame): Learning dataset.
            list_features (List[str] | None, optional): List of features. Defaults to None.
            list_cols_to_drop (List[str] | None, optional): List of columns to drop.
                Defaults to 'constants.LGBM_ID + [constants.LGBM_LABEL]'.
        """
        return super().define_features(df_learning, list_features, list_cols_to_drop)

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

        train_data = lgb.Dataset(df_train[self.features], label=df_train[self.target])
        valid_data = lgb.Dataset(df_valid[self.features], label=df_valid[self.target])
        lgbm_model = lgb.train(
            params=constants.LGBM_PARAMS,
            train_set=train_data,
            valid_sets=[train_data, valid_data],
        )
        self.model = lgbm_model
        logging.info("Training of the model completed")

    def predict(
        self: _ClassificationModel,
        df_to_predict: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict the label for a given dataset.

        Args:
            self (_ClassificationModel): Class object.
            df_to_predict (pd.DataFrame): Dataset to make predictions from.

        Returns:
            np.ndarray: Predictions.
        """
        df_to_predict = df_to_predict[self.features]
        predictions = self.model.predict(df_to_predict)
        return predictions

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

    def save_model(
        self: _ClassificationModel,
    ) -> None:
        """
        Save the model.

        Args:
            self (_ClassificationModel): Class object.
        """

    @classmethod
    def load_model(
        cls,
        name: str,
        path: str,
    ) -> _ClassificationModel:
        """
        Load previously trained model.

        Args:
            name (str): Name of the model.
            path (str): _description_

        Returns:
            _ClassificationModel: _description_
        """
        return pkl.load(open(os.path.join(path, name), "rb"), encoding="latin1")
