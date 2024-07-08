"""Names"""

###############################################################
#                                                             #
#                           DATA PATH                         #
#                                                             #
###############################################################

DATA_FOLDER = "data"
INPUT_FOLDER = "input"
RESULT_FOLDER = "result"

###############################################################
#                                                             #
#                           DATASETS                          #
#                                                             #
###############################################################

Y_TRAIN = "Y_train_1rknArQ.csv"


###############################################################
#                                                             #
#                        FREQUENT NAMES                       #
#                                                             #
###############################################################

GOALKEEPER = "goalkeeper"
DEFENDER = "defender"
MIDFIELDER = "midfielder"
ATTACKER = "attacker"


###############################################################
#                                                             #
#                        COLUMN NAMES                         #
#                                                             #
###############################################################

### COMMON

# IDs
ID = "id"
LEAGUE = "league"
TEAM_NAME = "team_name"
# VARIATIONS
SEASON = "season"
LAST_MATCHES = "5_last_match"
SUM = "sum"
AVERAGE = "average"
STD = "std"

### LABELS
# BASIS
HOME_WINS = "home_wins"
DRAW = "draw"
AWAY_WINS = "away_wins"
# CREATED
LABEL = "label"


### TEAM
# STATISTICS
TEAM_SHOTS_TOTAL = "team_shots_total"
TEAM_SHOTS_INSIDEBOX = "team_shots_insidebox"
TEAM_SHOTS_OFF_TARGET = "team_shots_off_target"
TEAM_SHOTS_ON_TARGET = "team_shots_on_target"
TEAM_SHOTS_OUTSIDEBOX = "team_shots_outsidebox"
TEAM_PASSES = "team_passes"
TEAM_SUCCESSFUL_PASSES = "team_successful_passes"
TEAM_SAVES = "team_saves"
TEAM_CORNERS = "team_corners"
TEAM_FOULS = "team_fouls"
TEAM_YELLOWCARDS = "team_yellowcards"
TEAM_REDCARDS = "team_redcards"
TEAM_OFFSIDES = "team_offsides"
TEAM_ATTACKS = "team_attacks"
TEAM_PENALTIES = "team_penalties"
TEAM_SUBSTITUTIONS = "team_substitutions"
TEAM_BALL_SAFE = "team_ball_safe"
TEAM_DANGEROUS_ATTACKS = "team_dangerous_attacks"
TEAM_INJURIES = "team_injuries"
TEAM_GOALS = "team_goals"
TEAM_GAME_WON = "team_game_won"
TEAM_GAME_DRAW = "team_game_draw"
TEAM_GAME_LOST = "team_game_lost"


### PLAYER
# METADATA
POSITION = "position"
PLAYER_NAME = "player_name"
# STATISTICS
PLAYER_ACCURATE_CROSSES = "player_accurate_crosses"
PLAYER_ACCURATE_PASSES = "player_accurate_passes"
PLAYER_AERIALS_WON = "player_aerials_won"
PLAYER_ASSISTS = "player_assists"
PLAYER_BIG_CHANCES_CREATED = "player_big_chances_created"
PLAYER_BIG_CHANCES_MISSED = "player_big_chances_missed"
PLAYER_BLOCKED_SHOTS = "player_blocked_shots"
PLAYER_CAPTAIN = "player_captain"
PLAYER_CLEARANCES = "player_clearances"
PLAYER_CLEARANCE_OFFLINE = "player_clearance_offline"
PLAYER_DISPOSSESSED = "player_dispossessed"
PLAYER_DRIBBLED_ATTEMPTS = "player_dribbled_attempts"
PLAYER_DRIBBLED_PAST = "player_dribbled_past"
PLAYER_DUELS_LOST = "player_duels_lost"
PLAYER_DUELS_WON = "player_duels_won"
PLAYER_ERROR_LEAD_TO_GOAL = "player_error_lead_to_goal"
PLAYER_FOULS = "player_fouls"
PLAYER_FOULS_DRAWN = "player_fouls_drawn"
PLAYER_GOALKEEPER_GOALS_CONCEDED = "player_goalkeeper_goals_conceded"
PLAYER_GOALS = "player_goals"
PLAYER_GOALS_CONCEDED = "player_goals_conceded"
PLAYER_HIT_WOODWORK = "player_hit_woodwork"
PLAYER_INTERCEPTIONS = "player_interceptions"
PLAYER_KEY_PASSES = "player_key_passes"
PLAYER_MINUTES_PLAYED = "player_minutes_played"
PLAYER_OFFSIDES = "player_offsides"
PLAYER_OWN_GOALS = "player_own_goals"
PLAYER_PASSES = "player_passes"
PLAYER_PENALTIES_COMMITTED = "player_penalties_committed"
PLAYER_PENALTIES_MISSES = "player_penalties_misses"
PLAYER_PENALTIES_SAVED = "player_penalties_saved"
PLAYER_PENALTIES_SCORED = "player_penalties_scored"
PLAYER_PENALTIES_WON = "player_penalties_won"
PLAYER_REDCARDS = "player_redcards"
PLAYER_SAVES = "player_saves"
PLAYER_SAVES_INSIDE_BOX = "player_saves_inside_box"
PLAYER_SHOTS_BLOCKED = "player_shots_blocked"
PLAYER_SHOTS_ON_TARGET = "player_shots_on_target"
PLAYER_SHOTS_TOTAL = "player_shots_total"
PLAYER_STARTING_LINEUP = "player_starting_lineup"
PLAYER_SUCCESSFUL_DRIBBLES = "player_successful_dribbles"
PLAYER_TACKLES = "player_tackles"
PLAYER_TOTAL_CROSSES = "player_total_crosses"
PLAYER_TOTAL_DUELS = "player_total_duels"
PLAYER_YELLOWCARDS = "player_yellowcards"
PLAYER_PUNCHES = "player_punches"
PLAYER_LONG_BALLS = "player_long_balls"
PLAYER_LONG_BALLS_WON = "player_long_balls_won"
PLAYER_SHOTS_OFF_TARGET = "player_shots_off_target"
# CREATED
PLAYER_RANK = "player_rank"
PLAYER_TO_KEEP = "player_to_keep"
