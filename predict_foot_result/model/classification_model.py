"""Abstract class for Classification"""

import abc
import typing
from typing import Tuple, Dict, List
import os
import pickle as pkl
import heapq
import numpy as np
import pandas as pd
from logger import logging


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from predict_foot_result.configs import names, constants


_ClassificationModel = typing.TypeVar(
    "_ClassificationModel", bound="ClassificationModel"
)


class ClassificationModel(abc.ABC):
    """
    Abstract class to represent classification models.

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(
        self: _ClassificationModel,
        name: str,
        path: str | None = None,
        target: str | None = None,
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
            path (str | None, optional): Path to the model. Defaults to None.
            target (str | None, optional): Target of the model. Defaults to None.
            features (list | str | None, optional): Features of the model. Defaults to None.
            params (dict | None, optional): Hyperparameters of the model. Defaults to None.
            cols_id (list | str | None, optional): Columns used as IDs. Defaults to None.
            train_valid_split (float, optional): Train-validation split.
                Defaults to constants.TRAIN_VALID_SPLIT.
            metrics (dict | None, optional): Metrics to evaluate the model. Defaults to None.
        """
        self.name = name
        self.path = path
        self.model = None
        self.target = target
        self.features = features
        self.params = params
        self.cols_id = cols_id
        self.scores = None
        self.train_valid_split = train_valid_split
        self.metrics = metrics
        return None

    def get_train_valid_sets(
        self: _ClassificationModel,
        df_learning: pd.DataFrame,
        is_temporal: bool = False,
        col_date: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and validation datasets.

        Args:
            self (_ClassificationModel): Class object.
            df_learning (pd.DataFrame): Learning set.
            is_temporal (bool, optional): Whether we face time series or not. Defaults to False.
            col_date (str | None, optional): Date column if temporal data. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation sets.
        """
        if is_temporal:
            valid_dates = heapq.nlargest(
                int(self.train_valid_split * df_learning[col_date].nunique()),
                df_learning[col_date].unique(),
            )
            df_train = df_learning[
                ~df_learning[col_date].isin(valid_dates)
            ].sort_values(self.cols_id)
            df_valid = df_learning[df_learning[col_date].isin(valid_dates)].sort_values(
                self.cols_id
            )
        else:
            df_train, df_valid = train_test_split(
                df_learning,
                test_size=self.train_valid_split,
                random_state=constants.RANDOM_SEED,
            )
            df_train = pd.DataFrame(df_train).sort_values(self.cols_id)
            df_valid = pd.DataFrame(df_valid).sort_values(self.cols_id)
        logging.info("Training and validation sets computed")
        return df_train, df_valid

    def define_features(
        self: _ClassificationModel,
        df_learning: pd.DataFrame,
        list_features: List[str] | None = None,
        list_cols_to_drop: List[str] | None = None,
    ) -> None:
        """
        Define features for the model.

        Args:
            self (_ClassificationModel): Class object.
            df_learning (pd.DataFrame): Learning dataset.
            list_features (List[str] | None, optional): List of features. Defaults to None.
            list_cols_to_drop (List[str] | None, optional): List of columns to drop. Defaults to None.
        """
        if list_features is not None:
            self.features = list_features
        elif list_cols_to_drop is not None:
            self.features = [
                col for col in df_learning.columns if col not in list_cols_to_drop
            ]
        else:
            self.features = df_learning.columns.tolist()
        return None

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    def score(
        self: _ClassificationModel,
        label_true: np.ndarray,
        label_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Get scores of the model.

        Args:
            self (_ClassificationModel): Class object.
            label_true (np.ndarray): True labels.
            label_pred (np.ndarray): Predicted labels.
        """
        accuracy = accuracy_score(label_true, label_pred)
        precision = precision_score(label_true, label_pred, average="macro")
        recall = recall_score(label_true, label_pred, average="macro")
        f1 = f1_score(label_true, label_pred, average="macro")
        dict_scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        logging.info(
            f"Accuracy: { accuracy }, Precision: { precision }, Recall: { recall }, F1: { f1 }"
        )
        return dict_scores

    def get_metrics(
        self: _ClassificationModel,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        """
        Get metrics of the model on train and valid sets.

        Args:
            self (_ClassificationModel): Class object.
            df_train (pd.DataFrame): Training set.
            df_valid (pd.DataFrame): Validation set.
        """
        # Get actual and predicted labels
        y_train_true = df_train[self.target]
        y_train_pred = self.predict(df_train)
        y_valid_true = df_valid[self.target]
        y_valid_pred = self.predict(df_valid)
        logging.info("Metrics on the training set:")
        train_scores = self.score(y_train_true, y_train_pred)
        logging.info("Metrics on the validation set:")
        valid_scores = self.score(y_valid_true, y_valid_pred)
        # Save the metrics
        dict_metrics = {
            "train": train_scores,
            "valid": valid_scores,
        }
        self.metrics = dict_metrics

    @abc.abstractmethod
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

    def training_pipeline(
        self: _ClassificationModel,
        df_learning: pd.DataFrame,
        fine_tuning: bool = False,
    ) -> None:
        """
        Pipeline for training and evaluating the model.

        Args:
            self (_ClassificationModel): Class object.
            df_learning (pd.DataFrame): Learning dataset.
            fine_tuning (bool, optional): whether fine-tuning the model. Defaults to False.
        """
        self.define_features(df_learning)
        df_train, df_valid = self.get_train_valid_sets(df_learning)
        if fine_tuning:
            self.fine_tune(df_train, df_valid)
        self.train(df_train, df_valid)
        self.get_metrics(df_train, df_valid)
        logging.info("Training pipeline completed")

    @abc.abstractmethod
    def save_model(
        self: _ClassificationModel,
    ) -> None:
        """
        Save the model.

        Args:
            self (_ClassificationModel): Class object.
        """

    @classmethod
    @abc.abstractmethod
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
