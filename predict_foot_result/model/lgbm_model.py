"""Class for LGBM model"""

from typing import List
import lightgbm as lgb
from logger import logging
import optuna
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
        features: list | str | None = None,
        metrics: dict | None = None,
    ) -> None:
        """
        Initialize class object.

        Args:
            self (_ClassificationModel): Class object.
            name (str): Name of the model.
            features (list | str | None, optional): Features of the model. Defaults to None.
            metrics (dict | None, optional): Metrics to evaluate the model. Defaults to None.
        """
        super().__init__(
            name=name, features=features, params=constants.LGBM_PARAMS, metrics=metrics
        )

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
            list_cols_to_drop (List[str] | None, optional): List of columns to drop.
                Defaults to None.
        """
        if list_cols_to_drop is None:
            list_cols_to_drop = constants.LGBM_ID + [constants.LGBM_LABEL]
        return super().define_features(
            df_learning=df_learning,
            list_features=list_features,
            list_cols_to_drop=list_cols_to_drop,
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
        train_data = lgb.Dataset(df_train[self.features], label=df_train[self.target])
        valid_data = lgb.Dataset(df_valid[self.features], label=df_valid[self.target])
        lgbm_model = lgb.train(
            params=self.params,
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
        predictions_proba = self.model.predict(df_to_predict)
        predictions_label = np.argmax(predictions_proba, axis=1)
        return predictions_label

    def fine_tuning_objective(
        self: _ClassificationModel,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        trial: optuna.Trial,
    ) -> float:
        """
        Objective function for fine-tuning hyperparameters using Optuna.

        Args:
            self (_ClassificationModel): Class object.
            df_train (pd.DataFrame): Training set.
            df_valid (pd.DataFrame): Validation set.
            trial (optuna.Trial): Trial for optimization.

        Returns:
            float: Metric value to optimize.
        """
        params = self.params.copy()
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
        params["num_leaves"] = trial.suggest_int("num_leaves", 20, 150)
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 10, 100)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
        params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.4, 1.0)
        params["lambda_l1"] = trial.suggest_float("lambda_l1", 0.01, 1.0)
        train_data = lgb.Dataset(df_train[self.features], label=df_train[self.target])
        valid_data = lgb.Dataset(df_valid[self.features], label=df_valid[self.target])
        lgbm_model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, valid_data],
        )
        self.model = lgbm_model
        accuracy = self.score(df_valid[self.target], self.predict(df_valid))["accuracy"]
        return accuracy

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
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.fine_tuning_objective(df_train, df_valid, trial),
            n_trials=constants.NB_OPTUNA_TRIALS,
        )
        self.params.update(study.best_params)
