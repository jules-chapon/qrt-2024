"""Data preprocessing"""

import os
from typing import List
from logger import logging
import numpy as np
import pandas as pd

from predict_foot_result.configs import names, constants


###############################################################
#                                                             #
#                        ALL FEATURES                         #
#                                                             #
###############################################################


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
    logging.info("Dataset of features imported")
    logging.info(f"{ df.shape[0] } rows and { df.shape[1] } columns loaded")
    return df


def rename_columns_for_home_or_away(
    df_features: pd.DataFrame, is_home: bool
) -> pd.DataFrame:
    """
    Rename columns to split between home and away features.

    Args:
        df_features (pd.DataFrame): dataset of features.
        is_home (bool): whether the dataset is for home or away.

    Returns:
        pd.DataFrame: datset with renamed columns.
    """
    if is_home:
        home = "home"
    else:
        home = "away"
    # Rename all columns except IDs
    df_features.set_index(names.ID, drop=True, inplace=True)
    df_features.columns = [f"{ home }_{ col }" for col in df_features.columns]
    df_features.reset_index(drop=False, inplace=True)
    return df_features


def merge_datasets(
    list_datasets: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge datasets.

    Args:
        list_datasets (List[pd.DataFrame]): list of datasets to merge.

    Returns:
        pd.DataFrame: merged dataset.
    """
    df_merged = list_datasets[0].copy()
    for df in list_datasets[1:]:
        df_merged = pd.merge(df_merged, df, on=names.ID, how="left")
    logging.info("Datasets merged")
    return df_merged


###############################################################
#                                                             #
#                        TEAM FEATURES                        #
#                                                             #
###############################################################


def keep_only_relevant_team_features(df_team_features: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only relevant features from the team dataset.

    Args:
        df_team_features (pd.DataFrame): dataset of team features.

    Returns:
        pd.DataFrame: dataset with the relevant team features.
    """
    # Define columns to keep
    cols_to_keep = (
        [names.ID]
        + [
            f"{ col }_{ names.SEASON }_{ names.SUM }"
            for col in constants.COLS_TEAM_FEATURES
        ]
        + [
            f"{ col }_{ names.LAST_MATCHES }_{ names.SUM }"
            for col in constants.COLS_TEAM_FEATURES
        ]
    )
    # Keep only the relevant columns
    df_team_relevant_features = df_team_features[cols_to_keep].copy()
    return df_team_relevant_features


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
    df_team_features = keep_only_relevant_team_features(
        df_team_features=df_team_features
    )
    df_team_features = rename_columns_for_home_or_away(
        df_features=df_team_features, is_home=is_home
    )
    return df_team_features


###############################################################
#                                                             #
#                       PLAYER FEATURES                       #
#                                                             #
###############################################################


def rank_players_by_position_and_appearances(
    df_player_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank players by minutes played for each possition and team.

    Args:
        df_player_features (pd.DataFrame): dataset of player features.

    Returns:
        pd.DataFrame: dataset with the new feature.
    """
    df_player_features[names.PLAYER_RANK] = df_player_features.groupby(
        constants.COLS_GROUPBY_PLAYER
    )["_".join([names.PLAYER_MINUTES_PLAYED, names.SEASON, names.SUM])].rank(
        method="dense"
    )
    return df_player_features


def keep_only_relevant_players(df_player_features: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only relevant players for each position and team.

    Args:
        df_player_features (pd.DataFrame): dataset of player features.

    Returns:
        pd.DataFrame: dataset with only relevant players.
    """
    df_player_features[names.RANK_THRESHOLD] = df_player_features[names.POSITION].map(
        constants.dict_nb_players_by_position
    )
    df_player_features[names.PLAYER_TO_KEEP] = np.where(
        df_player_features[names.PLAYER_RANK]
        <= df_player_features[names.RANK_THRESHOLD],
        True,
        False,
    )
    df_with_relevant_players = df_player_features[
        df_player_features[names.PLAYER_TO_KEEP] == 1
    ].copy()
    df_with_relevant_players.drop(
        columns=[names.RANK_THRESHOLD, names.PLAYER_TO_KEEP], inplace=True
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
        index=names.ID,
        columns=[names.POSITION, names.PLAYER_RANK],
        values=cols_stats,
        aggfunc="sum",
    )
    df_new_format.columns = [
        f"{col[1]}_{col[2]}_{col[0]}" if col[1] != "" else col[0]
        for col in df_new_format.columns
    ]
    return df_new_format


def keep_only_relevant_player_features(
    df_player_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only relevant player features.

    Args:
        df_player_features (pd.DataFrame): dataset of player features.
    Returns:
        pd.DataFrame: dataset with only relevant player features.
    """
    # Defining columns to keep
    cols_to_keep = []
    for position in constants.dict_nb_players_by_position.keys():
        for rank in range(constants.dict_nb_players_by_position[position]):
            cols_to_add_season = [
                f"{ position }_{ rank + 1 }_{ col_name }_{ names.SEASON }_{ names.SUM }"
                for col_name in constants.dict_relevant_features_by_position[position]
            ]
            cols_to_add_last_matches = [
                f"{ position }_{ rank + 1 }_{ col_name }_{ names.LAST_MATCHES }_{ names.SUM }"
                for col_name in constants.dict_relevant_features_by_position[position]
            ]
            cols_to_keep += cols_to_add_season + cols_to_add_last_matches
    # Selecting only relevant columns
    df_player_relevant_features = df_player_features[cols_to_keep].copy()
    df_player_relevant_features.reset_index(drop=False, inplace=True)
    return df_player_relevant_features


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
    df_player_features = keep_only_relevant_player_features(df_player_features)
    df_player_features = rename_columns_for_home_or_away(df_player_features, is_home)
    return df_player_features


###############################################################
#                                                             #
#                            LABELS                           #
#                                                             #
###############################################################


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
    ).astype(int)
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


###############################################################
#                                                             #
#                          PIPELINES                          #
#                                                             #
###############################################################


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
    df_learning = merge_datasets(
        [
            df_learning_labels,
            df_learning_team_home,
            df_learning_team_away,
            df_learning_player_home,
            df_learning_player_away,
        ]
    )
    return df_learning


def preprocessing_testing() -> pd.DataFrame:
    """
    Pipeline for the preprocessing of the testing dataset.

    Returns:
        pd.DataFrame: testing dataset.
    """
    df_testing_team_home = preprocessing_team_features(False, True)
    df_testing_team_away = preprocessing_team_features(False, False)
    df_testing_player_home = preprocessing_player_features(False, True)
    df_testing_player_away = preprocessing_player_features(False, False)
    df_testing = merge_datasets(
        [
            df_testing_team_home,
            df_testing_team_away,
            df_testing_player_home,
            df_testing_player_away,
        ]
    )
    return df_testing


# df_player_0 = import_features(True, True, False)
# df_player_1 = rank_players_by_position_and_appearances(df_player_0)
# df_player_2 = keep_only_relevant_players(df_player_1)
# df_player_3 = format_player_features(df_player_2)
# df_player_4 = keep_only_relevant_player_features(df_player_3)

# df_team_0 = import_features(True, True, True)
# df_team_1 = keep_only_relevant_team_features(df_team_0)

# df_features = merge_team_and_player_features(df_team_1, df_player_4)
