"""Data preprocessing"""

import os
from typing import List
import numpy as np
import pandas as pd

from predict_foot_result.configs import names, constants


def import_features(is_train: bool, is_home: bool, is_team: bool) -> pd.DataFrame:
    """
    Import dataset of features.

    Args:
        is_train (bool): whether the dataset is for training or testing.
        is_home (bool): whether the dataset is for home or away.
        is_team (bool): whether the dataset is for team or player stats.

    Returns:
        pd.DataFrame: dataset of features.
    """
    # Define characteristics of the dataset
    if is_train:
        train = "train"
    else:
        train = "test"
    if is_home:
        home = "home"
    else:
        home = "away"
    if is_team:
        team = "team"
    else:
        team = "player"
    # Import the given dataset
    file_name = f"{ train }_{ home }_{ team }_statistics_df.csv"
    csv_file_path = os.path.join(names.DATA_FOLDER, names.INPUT_FOLDER, file_name)
    df = pd.read_csv(csv_file_path)
    df.columns = [col.lower() for col in df.columns]
    return df


def keep_only_relevant_team_features(
    df_team_features: pd.DataFrame, is_home: bool
) -> pd.DataFrame:
    """
    Keep only relevant features from the team dataset.

    Args:
        df_team_features (pd.DataFrame): dataset of team features.
        is_home (bool): whether the dataset is for home or away team.

    Returns:
        pd.DataFrame: dataset with the relevant team features.
    """
    if is_home:
        home = "home"
    else:
        home = "away"
    # Define columns to keep
    cols_to_keep = (
        constants.COLS_FEATURES_ID
        + [
            f"{ home }_{ col }_{ names.SEASON }_{ names.SUM }"
            for col in constants.COLS_TEAM_FEATURES
        ]
        + [
            f"{ home }_{ col }_{ names.LAST_MATCHES }_{ names.SUM }"
            for col in constants.COLS_TEAM_FEATURES
        ]
    )
    # Keep only the relevant columns
    df_team_relevant_features = df_team_features[cols_to_keep].copy()
    return df_team_relevant_features


def rank_players_by_position_and_appearances(
    df_player_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank players by appearances for each possition and team.

    Args:
        df_player_features (pd.DataFrame): dataset of player features.

    Returns:
        pd.DataFrame: dataset with the new feature.
    """
    cols_groupby = [names.ID, names.TEAM_NAME, names.POSITION]
    df_player_features[names.PLAYER_RANK] = df_player_features.groupby(cols_groupby)[
        "_".join([names.PLAYER_STARTING_LINEUP, names.SEASON, names.SUM])
    ].rank(method="dense")
    return df_player_features


def keep_only_relevant_players(df_player_features: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only relevant players for each position and team.

    Args:
        df_player_features (pd.DataFrame): dataset of player features.

    Returns:
        pd.DataFrame: dataset with only relevant players.
    """
    df_player_features["rank_threshold"] = df_player_features[names.POSITION].map(
        constants.dict_nb_players_by_position
    )
    df_player_features[names.PLAYER_TO_KEEP] = np.where(
        df_player_features[names.PLAYER_RANK] <= df_player_features["rank_threshold"],
        True,
        False,
    )
    df_with_relevant_players = df_player_features[
        df_player_features[names.PLAYER_TO_KEEP] == 1
    ].copy()
    df_with_relevant_players.drop(
        columns=["rank_threshold", names.PLAYER_TO_KEEP], inplace=True
    )
    df_with_relevant_players[names.PLAYER_RANK] = df_with_relevant_players[
        names.PLAYER_RANK
    ].astype(int)
    return df_with_relevant_players


def format_player_features(df_player_features: pd.DataFrame) -> pd.DataFrame:
    """
    Format dataset features to make them easier to use.
    We switch to a wide format.

    Args:
        df_player_features (pd.DataFrame): dataset of player features.

    Returns:
        pd.DataFrame: dataset of player features in the new format.
    """
    cols_stats = [
        col
        for col in df_player_features.columns
        if col
        not in constants.COLS_PLAYER_CATEGORICAL
        + [names.PLAYER_TO_KEEP, names.PLAYER_RANK]
    ]
    df_new_format = pd.pivot_table(
        df_player_features,
        index=[names.ID, names.TEAM_NAME],
        columns=[names.POSITION, names.PLAYER_RANK],
        values=cols_stats,
        aggfunc="sum",
    )
    df_new_format.columns = [
        f"{col[1]}_{col[2]}_{col[0]}" if col[1] != "" else col[0]
        for col in df_new_format.columns
    ]
    return df_new_format


def preprocessing_player_features(
    is_train: bool,
    is_home: bool,
) -> pd.DataFrame:
    """
    Preprocessing of player features.

    Args:
        is_train (bool): whether the dataset is for training or testing.
        is_home (bool): whether the dataset is for home or away.

    Returns:
        pd.DataFrame: clean dataset of player features.
    """
    df_player_features = import_features(
        is_train=is_train, is_home=is_home, is_team=False
    )
    df_player_features = rank_players_by_position_and_appearances(df_player_features)
    df_player_features = keep_only_relevant_players(df_player_features)
    df_player_features = format_player_features(df_player_features)
    return df_player_features


def keep_only_relevant_player_features(
    df_player_features: pd.DataFrame,
    is_home: bool,
) -> pd.DataFrame:
    """
    Keep only relevant player features.

    Args:
        df_player_features (pd.DataFrame): dataset of player features.
        is_home (bool): whether the dataset is for home or away team.

    Returns:
        pd.DataFrame: dataset with only relevant player features.
    """
    if is_home:
        home = "home"
    else:
        home = "away"
    # Defining columns to keep
    cols_to_keep = []
    for position in constants.dict_nb_players_by_position.keys():
        for rank in range(constants.dict_nb_players_by_position[position]):
            cols_to_add_season = [
                f"{ home }_{ position }_{ rank + 1 }_{ col_name }_{ names.SEASON }_{ names.SUM }"
                for col_name in constants.dict_relevant_features_by_position[position]
            ]
            cols_to_add_last_matches = [
                f"{ home }_{ position }_{ rank + 1 }_{ col_name }_{ names.LAST_MATCHES }_{ names.SUM }"
                for col_name in constants.dict_relevant_features_by_position[position]
            ]
            cols_to_keep += cols_to_add_season + cols_to_add_last_matches
    # Selecting only relevant columns
    df_player_relevant_features = df_player_features[cols_to_keep].copy()
    df_player_relevant_features.reset_index(drop=False, inplace=True)
    return df_player_relevant_features


def preprocessing_team_features(is_train: bool, is_home: bool) -> pd.DataFrame:
    """
    Preprocessing of team features.

    Args:
        is_train (bool): whether the dataset is for training or testing.
        is_home (bool): whether the dataset is for home or away team.

    Returns:
        pd.DataFrame: clean dataset of team features.
    """
    df_team_features = import_features(is_train=is_train, is_home=is_home, is_team=True)
    df_team_features = keep_only_relevant_team_features(df_team_features, is_home)
    return df_team_features


def merge_features(
    list_df_features: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge datasets of features.

    Args:
        list_df_features (List[pd.DataFrame]): list of datasets of features.

    Returns:
        pd.DataFrame: dataset with all features.
    """
    df_features = list_df_features[0]
    for df in list_df_features[1:]:
        df_features = df_features.merge(df, on=constants.COLS_FEATURES_ID, how="left")
    return df_features


def import_training_labels() -> pd.DataFrame:
    """
    Import training dataset of labels.

    Returns:
        pd.DataFrame: dataset with labels.
    """
    # Import the dataset
    csv_file_path = os.path.join(names.DATA_FOLDER, names.INPUT_FOLDER, names.Y_TRAIN)
    df = pd.read_csv(csv_file_path)
    df.columns = [col.lower() for col in df.columns]
    return df


def clean_training_labels(df_labels: pd.DataFrame) -> pd.DataFrame:
    df_labels[names.LABEL] = np.where(
        df_labels[names.HOME_WINS] == 1,
        0,
        np.where(
            df_labels[names.DRAW] == 1,
            1,
            np.where(df_labels[names.AWAY_WINS] == 1, 2, np.nan),
        ),
    )
    df_labels.drop(columns=[names.HOME_WINS, names.DRAW, names.AWAY_WINS], inplace=True)
    return df_labels


def preprocessing_labels() -> pd.DataFrame:
    """
    Preprocessing labels.

    Returns:
        pd.DataFrame: clean dataset of labels.
    """
    df_labels = import_training_labels()
    df_labels = clean_training_labels(df_labels)
    return df_labels


def merge_labels_and_features(
    df_labels: pd.DataFrame, df_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge labels and features datasets.

    Args:
        df_labels (pd.DataFrame): dataset of labels.
        df_features (pd.DataFrame): dataset of features.

    Returns:
        pd.DataFrame: dataset with labels and features.
    """
    df_merged = pd.merge(df_labels, df_features, on=names.ID, how="left")
    return df_merged


def preprocessing_learning() -> pd.DataFrame:
    """
    Pipeline for the preprocessing of the learning dataset.

    Returns:
        pd.DataFrame: learning dataset.
    """
    df_learning_labels = preprocessing_labels()
    df_learning_team_home = preprocessing_team_features(True, True)
    df_learning_team_away = preprocessing_team_features(True, False)
    df_learning_player_home = preprocessing_player_features(True, True)
    df_learning_player_away = preprocessing_player_features(True, False)
    df_learning_features = merge_features(
        [
            df_learning_team_home,
            df_learning_team_away,
            df_learning_player_home,
            df_learning_player_away,
        ]
    )
    df_learning = merge_labels_and_features(df_learning_labels, df_learning_features)
    return df_learning


# df_player_0 = import_features(True, True, False)
# df_player_1 = rank_players_by_position_and_appearances(df_player_0)
# df_player_2 = keep_only_relevant_players(df_player_1)
# df_player_3 = format_player_features(df_player_2)
# df_player_4 = keep_only_relevant_player_features(df_player_3)

# df_team_0 = import_features(True, True, True)
# df_team_1 = keep_only_relevant_team_features(df_team_0)

# df_features = merge_team_and_player_features(df_team_1, df_player_4)
