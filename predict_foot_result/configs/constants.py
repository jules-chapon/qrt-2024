"""Constants"""

from predict_foot_result.configs import names


###############################################################
#                                                             #
#                        FIXED VALUES                         #
#                                                             #
###############################################################

RANDOM_SEED = 42


###############################################################
#                                                             #
#                           ML CONFIG                         #
#                                                             #
###############################################################

TRAIN_VALID_SPLIT = 0.2

TARGET = names.LABEL

###############################################################
#                                                             #
#                           LIGHTGBM                          #
#                                                             #
###############################################################

LGBM_LABEL = names.LABEL

LGBM_ID = [names.ID, names.HOME_TEAM_NAME, names.AWAY_TEAM_NAME]

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "random_seed": RANDOM_SEED,
    "verbose": -1,
    "num_estimators": 100,
    "early_stopping_rounds": 10,
    "learning_rate": 0.1,
    "max_depth": 15,
    "max_leaves": 31,
    "min_data_per_leaf": 20,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "feature_fraction": 0.8,
    # "lambda_l1": 0.1,
}

###############################################################
#                                                             #
#                       PREPROCESSING                         #
#                                                             #
###############################################################

dict_nb_players_by_position = {
    names.GOALKEEPER: 1,
    names.DEFENDER: 5,
    names.MIDFIELDER: 5,
    names.ATTACKER: 5,
}

dict_relevant_features_by_position = {
    names.GOALKEEPER: [
        names.PLAYER_MINUTES_PLAYED,
        names.PLAYER_FOULS,
        names.PLAYER_REDCARDS,
        names.PLAYER_ERROR_LEAD_TO_GOAL,
        names.PLAYER_GOALKEEPER_GOALS_CONCEDED,
        names.PLAYER_SAVES,
        names.PLAYER_SAVES_INSIDE_BOX,
        names.PLAYER_PUNCHES,
        names.PLAYER_PENALTIES_SAVED,
        names.PLAYER_ACCURATE_PASSES,
    ],
    names.DEFENDER: [
        names.PLAYER_MINUTES_PLAYED,
        names.PLAYER_FOULS,
        names.PLAYER_REDCARDS,
        names.PLAYER_ERROR_LEAD_TO_GOAL,
        names.PLAYER_TACKLES,
        names.PLAYER_DRIBBLED_PAST,
        names.PLAYER_INTERCEPTIONS,
        names.PLAYER_AERIALS_WON,
        names.PLAYER_BLOCKED_SHOTS,
        names.PLAYER_DUELS_WON,
        names.PLAYER_DUELS_LOST,
        names.PLAYER_ACCURATE_PASSES,
    ],
    names.MIDFIELDER: [
        names.PLAYER_MINUTES_PLAYED,
        names.PLAYER_FOULS,
        names.PLAYER_REDCARDS,
        names.PLAYER_ERROR_LEAD_TO_GOAL,
        names.PLAYER_DISPOSSESSED,
        names.PLAYER_TACKLES,
        names.PLAYER_DRIBBLED_PAST,
        names.PLAYER_INTERCEPTIONS,
        names.PLAYER_DUELS_WON,
        names.PLAYER_DUELS_LOST,
        names.PLAYER_ACCURATE_PASSES,
        names.PLAYER_ACCURATE_CROSSES,
        names.PLAYER_KEY_PASSES,
        names.PLAYER_SHOTS_ON_TARGET,
        names.PLAYER_SUCCESSFUL_DRIBBLES,
    ],
    names.ATTACKER: [
        names.PLAYER_MINUTES_PLAYED,
        names.PLAYER_FOULS,
        names.PLAYER_OFFSIDES,
        names.PLAYER_REDCARDS,
        names.PLAYER_TACKLES,
        names.PLAYER_INTERCEPTIONS,
        names.PLAYER_DUELS_WON,
        names.PLAYER_DUELS_LOST,
        names.PLAYER_KEY_PASSES,
        names.PLAYER_BIG_CHANCES_CREATED,
        names.PLAYER_BIG_CHANCES_MISSED,
        names.PLAYER_PENALTIES_SCORED,
        names.PLAYER_SHOTS_ON_TARGET,
        names.PLAYER_GOALS,
    ],
}


###############################################################
#                                                             #
#                           COLUMNS                           #
#                                                             #
###############################################################

COLS_GROUPBY_PLAYER = [names.ID, names.POSITION]

COLS_TEAM_FEATURES = [
    names.TEAM_SHOTS_TOTAL,
    names.TEAM_SHOTS_INSIDEBOX,
    names.TEAM_SHOTS_OFF_TARGET,
    names.TEAM_SHOTS_ON_TARGET,
    names.TEAM_SHOTS_OUTSIDEBOX,
    names.TEAM_PASSES,
    names.TEAM_SUCCESSFUL_PASSES,
    names.TEAM_SAVES,
    names.TEAM_CORNERS,
    names.TEAM_FOULS,
    names.TEAM_YELLOWCARDS,
    names.TEAM_REDCARDS,
    names.TEAM_OFFSIDES,
    names.TEAM_ATTACKS,
    names.TEAM_PENALTIES,
    names.TEAM_SUBSTITUTIONS,
    names.TEAM_BALL_SAFE,
    names.TEAM_DANGEROUS_ATTACKS,
    names.TEAM_INJURIES,
    names.TEAM_GOALS,
    names.TEAM_GAME_WON,
    names.TEAM_GAME_DRAW,
    names.TEAM_GAME_LOST,
]

COLS_PLAYER_CATEGORICAL = [
    names.ID,
    names.LEAGUE,
    names.TEAM_NAME,
    names.POSITION,
    names.PLAYER_NAME,
]
