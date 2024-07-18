"""Class for smart model"""

import optuna
import numpy as np
import pandas as pd

from predict_foot_result.configs import names, constants
from predict_foot_result.model.classification_model import (
    ClassificationModel,
    _ClassificationModel,
)


class SmartClassificationModel(ClassificationModel):
    """
    Class to represent smart classification models.

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

    def define_evaluation_metric_for_training(
        self: _ClassificationModel, df_train: pd.DataFrame, trial: optuna.Trial
    ) -> float:
        """
        Define evaluation metric for training.

        Args:
            self (_ClassificationModel): Class object.
            df_train (pd.DataFrame): Training set.
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Evaluation metric for training.
        """
        # Define parameters
        alpha = trial.suggest_uniform(names.ALPHA, -1, 1)
        beta = trial.suggest_uniform(names.BETA, -1, alpha)
        # Create predicted labels
        df_train[names.TEAM_DIFF_WINRATE] = (
            df_train[
                f"{ names.HOME }_{ names.TEAM_GAME_WON }_{ names.SEASON }_{ names.AVERAGE }"
            ]
            - df_train[
                f"{ names.AWAY }_{ names.TEAM_GAME_WON }_{ names.SEASON }_{ names.AVERAGE }"
            ]
        )
        df_train[names.PREDICTED_LABEL] = np.where(
            df_train[names.TEAM_DIFF_WINRATE] >= alpha,
            0,
            np.where(df_train[names.TEAM_DIFF_WINRATE] < beta, 2, 1),
        )
        # Compute accuracy
        accuracy = self.score(df_train[names.LABEL], df_train[names.PREDICTED_LABEL])[
            names.ACCURACY
        ]
        return -accuracy

    def train(
        self: _ClassificationModel,
        df_train: pd.DataFrame,
    ) -> None:
        """
        Train the model and store it to self.model.

        Args:
            self (_ClassificationModel): Class object.
            df_train (pd.DataFrame): Training set.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.define_evaluation_metric_for_training(df_train, trial),
            n_trials=constants.NB_OPTUNA_TRIALS,
        )
        self.params = study.best_params

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
        alpha = self.params[names.ALPHA]
        beta = self.params[names.BETA]
        df_to_predict[names.TEAM_DIFF_WINRATE] = (
            df_to_predict[
                f"{ names.HOME }_{ names.TEAM_GAME_WON }_{ names.SEASON }_{ names.AVERAGE }"
            ]
            - df_to_predict[
                f"{ names.AWAY }_{ names.TEAM_GAME_WON }_{ names.SEASON }_{ names.AVERAGE }"
            ]
        )
        df_to_predict[names.PREDICTED_LABEL] = np.where(
            df_to_predict[names.TEAM_DIFF_WINRATE] >= alpha,
            0,
            np.where(df_to_predict[names.TEAM_DIFF_WINRATE] < beta, 2, 1),
        )
        predictions_label = df_to_predict[names.PREDICTED_LABEL]
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
