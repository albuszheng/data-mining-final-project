# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np


# %%
match_file = pd.read_csv('dota-2-matches/match.csv')
match_file.head()


# %%
# details of bit string is in :
# https://wiki.teamfortress.com/wiki/WebAPI/GetMatchDetails#Player_Slot
def tower_status(ts_radiant, ts_dire):
    tsr = {}
    tsd = {}
    bit_tsr = '{0:016b}'.format(ts_radiant)
    bit_tsd = '{0:016b}'.format(ts_dire)
    tsr['top'] = bit_tsr.count('1', -3)
    tsd['top'] = bit_tsd.count('1', -3)
    tsr['mid'] = bit_tsr.count('1', 10, 13)
    tsd['mid'] = bit_tsd.count('1', 10, 13)
    tsd['bottom'] = bit_tsd.count('1', 7, 10)
    tsr['bottom'] = bit_tsr.count('1', 7, 10)
    tsd['ancient'] = bit_tsd.count('1', 5, 7)
    tsr['ancient'] = bit_tsr.count('1', 5, 7)
    return (tsr, tsd)

def barracks_status(bs_radiant, bs_dire):
    bsr = {}
    bsd = {}
    bit_bsr = '{0:08b}'.format(bs_radiant)
    bit_bsd = '{0:08b}'.format(bs_dire)
    bsr['top'] = bit_bsr.count('1', -2)
    bsd['top'] = bit_bsd.count('1', -2)
    bsr['mid'] = bit_bsr.count('1', 2, 4)
    bsd['mid'] = bit_bsd.count('1', 2, 4)
    bsd['bottom'] = bit_bsd.count('1', 4, 6)
    bsr['bottom'] = bit_bsr.count('1', 4, 6)
    return (bsr, bsd)


# %%
df_players = pd.read_csv(
    'dota-2-matches/players.csv', 
    usecols=[
        'match_id',
        'player_slot',
        'gold',
        'gold_spent',
        'kills',
        'deaths',
        'assists',
        'denies',
        'last_hits',
        'hero_damage',
        'tower_damage',
        'level',
        'gold_buyback'
        ])
# df_player_time = pd.read_csv('dota-2-matches/player_time.csv')
# df_ability = pd.read_csv('dota-2-matches/ability_upgrades.csv')
df_team_fights = pd.read_csv('dota-2-matches/teamfights.csv')
df_team_fights_players = pd.read_csv('dota-2-matches/teamfights_players.csv')


# %%
match_data = []
match_file = match_file[match_file['game_mode'] == 22]
df_players.fillna(0, inplace=True)

radiant_pl = [0,1,2,3,4]
dire_pl = [128,129,130,131,132]
player_features = {
    'gold': 'full_total',
    'gold_spent': 'full_avg',
    'kills': 'only_total',
    'deaths': 'full_total',
    'assists': 'full_avg',
    'denies': 'full_avg',
    'last_hits': 'full_avg',
    'hero_damage': 'full_total',
    'tower_damage': 'full_total',
    'level': 'full_total',
    'gold_buyback': 'full_avg'
    }


# %%
df = df_players[(df_players.player_slot.isin(radiant_pl)) & (df_players.match_id == 0)]
df


# %%
tf = df_team_fights_players[df_team_fights_players.match_id == 0]
# for i in list(range(0,int(len(tf)/10))):
#     print(i*10,(i+1)*10)
a = tf[0:10]
d = sum(a[a.player_slot.isin(dire_pl)]['deaths'])
d


# %%
def teamfight_result(teamfights):
    loss_d = 0
    loss_r = 0
    for i in list(range(0,int(len(tf)/10))):
        tf_df = teamfights[i*10:(i+1)*10]
        rd = sum(tf_df[tf_df.player_slot.isin(radiant_pl)]['deaths'])
        dd = sum(tf_df[tf_df.player_slot.isin(dire_pl)]['deaths'])  
        if dd < rd:
            loss_r += 1
        elif rd < dd:
            loss_d += 1
    return (loss_r, loss_d)


# %%
def stat_agg(types: str, feature_name: str, data_list: str, team_data: dict):
    if types == "only_total":
        team_data[f'{feature_name}_total'] = sum(data_list)
    elif types == "full_total":
        team_data[f'{feature_name}_total'] = sum(data_list)
        team_data[f'{feature_name}_max'] = max(data_list)
        team_data[f'{feature_name}_min'] = min(data_list)
        team_data[f'{feature_name}_std'] = round(np.std(data_list), 4)
    elif types == "full_avg":
        team_data[f'{feature_name}_avg'] = np.average(data_list)
        team_data[f'{feature_name}_max'] = max(data_list)
        team_data[f'{feature_name}_min'] = min(data_list)
        team_data[f'{feature_name}_std'] = round(np.std(data_list), 4)

    return team_data


# %%
def aggregation_data(match_id, team, team_data: dict):
    # getting the player list
    player_ids = radiant_pl if team == 'radiant' else dire_pl

    filter_players = (df_players.player_slot.isin(player_ids)) & (df_players.match_id == match_id)
    df_team_players = df_players[filter_players]
    
    for feature in player_features:
        team_data = stat_agg(player_features[feature], feature, df_team_players[feature], team_data)

    return team_data


# %%
for idx, row in match_file.iterrows():
    match_id = row['match_id']
    duration = row['duration']

    # Tower, barracks, ancient status
    tower_radiant, tower_dire = tower_status(row['tower_status_radiant'], row['tower_status_dire'])
    barracks_radiant, barracks_dire = barracks_status(row['barracks_status_radiant'], row['barracks_status_dire'])

    # teamfights result
    loss_radiant, loss_dire = teamfight_result(df_team_fights_players[df_team_fights_players.match_id == match_id])

    #-- radiant --#
    # init
    team_radiant = {'match_id': match_id, 'duration': duration}
    # result
    team_radiant['result'] = 1 if row['radiant_win'] else 0
    # tower, barrack, ancient comparison data
    team_radiant['top_towers'] = tower_radiant['top'] - tower_dire['top']
    team_radiant['mid_towers'] = tower_radiant['mid'] - tower_dire['mid']
    team_radiant['bottom_towers'] = tower_radiant['bottom'] - tower_dire['bottom']
    team_radiant['ancient_status'] = tower_radiant['ancient'] - tower_dire['ancient']
    team_radiant['top_barracks'] = barracks_radiant['top'] - barracks_dire['top']
    team_radiant['mid_barracks'] = barracks_radiant['mid'] - barracks_dire['mid']
    team_radiant['bottom_barracks'] = barracks_radiant['bottom'] - barracks_dire['bottom']
    # aggregating data from players, abilities
    team_radiant = aggregation_data(match_id, 'radiant', team_radiant)
    # teamfight
    team_radiant['teamfight_loss'] = loss_radiant

    #-- dire --#
    # init
    team_dire = {'match_id': match_id, 'duration': duration}
    # result
    team_dire['result'] = 0 if row['radiant_win'] else 1
    # tower, barrack, ancient comparison data
    team_dire['top_towers'] = - tower_radiant['top'] + tower_dire['top']
    team_dire['mid_towers'] = - tower_radiant['mid'] + tower_dire['mid']
    team_dire['bottom_towers'] = - tower_radiant['bottom'] + tower_dire['bottom']
    team_dire['ancient_status'] = - tower_radiant['ancient'] + tower_dire['ancient']
    team_dire['top_barracks'] = - barracks_radiant['top'] + barracks_dire['top']
    team_dire['mid_barracks'] = - barracks_radiant['mid'] + barracks_dire['mid']
    team_dire['bottom_barracks'] = - barracks_radiant['bottom'] + barracks_dire['bottom']
    # aggregating data from players, abilities
    team_dire = aggregation_data(match_id, 'dire', team_dire)
    # teamfight
    team_dire['teamfight_loss'] = loss_dire

    #-- summarize --#
    match_data.append(team_radiant)
    match_data.append(team_dire)


# %%
len(match_file)


# %%
len(match_data)


# %%
match_data[0]


# %%
df_match_data = pd.DataFrame(match_data)


# %%
df_match_data.to_csv('cleaned_match_data')

