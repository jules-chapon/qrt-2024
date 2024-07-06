"""Data preprocessing"""

import os
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


def clean_team_features(df_team_features: pd.DataFrame) -> pd.DataFrame:
    return None


def clean_player_features(df_player_features: pd.DataFrame) -> pd.DataFrame:
    return None


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
    df_player_features[names.PLAYER_TO_KEEP] = False
    for position in constants.dict_nb_players_by_position.keys():
        df_player_features[names.PLAYER_TO_KEEP] = np.where(
            (df_player_features[names.POSITION] == position)
            & (
                df_player_features[names.PLAYER_RANK]
                <= constants.dict_nb_players_by_position[position]
            ),
            True,
            False,
        )
    df_with_relevant_players = df_player_features[
        df_player_features[names.PLAYER_TO_KEEP] == 1
    ].copy()
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


def transform_player_features(df_player_features: pd.DataFrame) -> pd.DataFrame:
    return None


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
