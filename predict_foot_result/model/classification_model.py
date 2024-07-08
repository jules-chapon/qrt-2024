"""Abstract class for Classification"""

import abc
import typing
from typing import Tuple
import os
import pickle as pkl
import heapq
import pandas as pd


from sklearn.model_selection import train_test_split

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
        cols_id: list | str | None = None,
        train_valid_split: float = constants.TRAIN_VALID_SPLIT,
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
                random_state=constants.RANDOM_STATE,
            )
            df_train = pd.DataFrame(df_train).sort_values(self.cols_id)
            df_valid = pd.DataFrame(df_valid).sort_values(self.cols_id)
        return df_train, df_valid

    @abc.abstractmethod
    def train_model(
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

    @abc.abstractmethod
    def score(
        self: _ClassificationModel,
        df_to_score: pd.DataFrame,
    ) -> None:
        """
        Get scores of the model.

        Args:
            self (_ClassificationModel): Class object.
            df_to_score (pd.DataFrame): DataFrame with predicted and actual values.
        """

    @classmethod
    @abc.abstractmethod
    def fine_tune(
        cls,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        """
        Fine-tune hypermarameters of the model.

        Args:
            df_train (pd.DataFrame): Training set.
            df_valid (pd.DataFrame): Validation set.
        """

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
