"""Constants"""

from predict_foot_result.configs import names

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


###############################################################
#                                                             #
#                           COLUMNS                           #
#                                                             #
###############################################################

COLS_PLAYER_CATEGORICAL = [
    names.ID,
    names.LEAGUE,
    names.TEAM_NAME,
    names.POSITION,
    names.PLAYER_NAME,
]
