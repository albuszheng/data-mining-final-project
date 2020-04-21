# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import r2_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegressionCV

import xgboost as xgb
from xgboost import XGBClassifier

import lightgbm as gbm
from lightgbm import LGBMClassifier

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Data aggregation

# %%
match_file = pd.read_csv('dota-2-matches/match.csv')
match_file.head()


# %%
# Util function for data aggregation
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
df_team_fights = pd.read_csv('dota-2-matches/teamfights.csv')
df_team_fights_players = pd.read_csv('dota-2-matches/teamfights_players.csv')

# %% [markdown]
# ### Novel Features - negative chat
# 
# We tried a custom known list of reliably negative words in chat as a novel feature. We count the number of occurrences of each word in the dictionary in chat per team per match.

# %%
df_chat = pd.read_csv('dota-2-matches/chat.csv')
df_chat['key'].fillna('', inplace=True)

naughty_words = [
    'stfu',
    'ez',
    'fuck',
    'wtf',
    'blame',
    'report',
    'reported',
    'shit',
    'ass',
    'asshole',
    'idiot',
    'stupid',
    'support',
    'blyat',
    'noob',
    'gg'
]

def get_naughty_count(phrase):
    naughty_count = 0
    tokens = phrase.split()
    for token in tokens:
        naughty_count = naughty_count + (1 if token in naughty_words else 0)
    return naughty_count

df_chat['is_radiant'] = df_chat['slot'] < 5
df_chat['naughty_count'] = df_chat['key'].apply(get_naughty_count)
df_chat.head()


# %%
df_chat['naughty_count'].describe()

# %% [markdown]
# We can see that the median negative word count is 0, and the majority of games have 0 instances of negative words. Thus, later we convert it to a binary feature (present or not).

# %%
df_naughty_count_only = pd.DataFrame(df_chat['naughty_count'], columns=['naughty_count'])
df_naughty_count_only.hist()


# %%
df_match_grouped = df_chat.groupby(['match_id', 'is_radiant'], as_index=False).naughty_count.agg('sum')
df_match_grouped['radiant_naughty_count'] = np.where(df_match_grouped['is_radiant'] == True, df_match_grouped['naughty_count'], 0)
df_match_grouped['dire_naughty_count'] = np.where(df_match_grouped['is_radiant'] == False, df_match_grouped['naughty_count'], 0)
df_match_grouped.head()


# %%
df_match_naughty_counts = df_match_grouped.groupby(['match_id']).agg({
    'radiant_naughty_count': 'sum',
    'dire_naughty_count': 'sum'
})

df_match_naughty_counts.head()


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

# %% [markdown]
# ### Novel Feature - Teamfight result

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
df_players.dtypes


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
tf = df_team_fights_players[df_team_fights_players.match_id == 0]

# %% [markdown]
# ### Final Data Aggregation

# %%
for idx, row in match_file.iterrows():
    match_id = row['match_id']
    duration = row['duration']

    # Tower, barracks, ancient status
    tower_radiant, tower_dire = tower_status(row['tower_status_radiant'], row['tower_status_dire'])
    barracks_radiant, barracks_dire = barracks_status(row['barracks_status_radiant'], row['barracks_status_dire'])

    # teamfights result
    loss_radiant, loss_dire = teamfight_result(df_team_fights_players[df_team_fights_players.match_id == match_id])

    # naughty word count
    naughty_counts = None
    try: 
        naughty_counts = df_match_naughty_counts.loc[match_id]
    except:
        pass

    radiant_naughty_count = 0
    dire_naughty_count = 0

    radiant_naughty_count = naughty_counts['radiant_naughty_count'] if naughty_counts is not None else 0
    dire_naughty_count = naughty_counts['radiant_naughty_count'] if naughty_counts is not None else 0

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
    # naughty count
    team_radiant['has_negative_chat'] = True if radiant_naughty_count > 0 else False

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
    # naughty word count
    team_dire['has_negative_chat'] = True if dire_naughty_count > 0 else False

    match_data.append(team_radiant)
    match_data.append(team_dire)


# %%
# Permenet storage for the cleaned and aggregated data
df_match_data = pd.DataFrame(match_data)
df_match_data.to_csv('data_clean/cleaned_match_data.csv')

# %% [markdown]
# # Model Building
# %% [markdown]
# ### Util function for model training

# %%
# util function for plot building

def plot_roc_curve(y_train, preds_train, y_test, preds_test):
    plt.plot(metrics.roc_curve(y_train, preds_train)[0], metrics.roc_curve(y_train, preds_train)[1],
             color = 'red', label='Train ROC Curve (area = %0.5f)' % roc_auc_score(y_train, preds_train))
    plt.plot(metrics.roc_curve(y_test, preds_test)[0],metrics.roc_curve(y_test, preds_test)[1],
             color = 'blue', label='Test ROC Curve (area = %0.5f)' % roc_auc_score(y_test, preds_test))
    plt.plot([0, 2], [0, 2], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend()
    plt.show()
    sns.set(style='white', rc={'figure.figsize':(10,10)})


# %%
def important_stats(y_true, y_pred_proba, summary):
    print("------------------------------------------")
    y_pred_label = pd.Series(y_pred_proba)
    y_pred_label = y_pred_label.map(lambda x: 1 if x > 0.5 else 0)
    print(summary)
    reacll = recall_score(y_true, y_pred_label)
    print('recall:', reacll)
    f1_stat = f1_score(y_true, y_pred_label)
    print('f1_score:', f1_stat)
    accuracyScore= accuracy_score(y_true, y_pred_label)
    print('accuracy_score:', accuracyScore)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_proba)
    auc = metrics.auc(fpr, tpr)
    print('AUC:', auc)
    matrix = pd.crosstab(y_true, y_pred_label, rownames=['True'], colnames=['Predicted'], margins=True)
    print(matrix)
    print("------------------------------------------")

# %% [markdown]
# ### Loading the data set and initialization

# %%
df = pd.read_csv(f'{os.getcwd()}/data_clean/cleaned_match_data.csv')


# %%
df.head()


# %%
df.shape


# %%
df.dtypes


# %%
df.head()
df = df.drop(columns = ['match_id','Unnamed: 0'])


# %%
df.corr()


# %%
#Drop highly correlated features
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column]>0.6)]
df.drop(to_drop,axis=1,inplace=True)
df.head()


# %%
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns = ['result','duration']),
    df['result'],
    test_size=0.2,
    random_state=1
)


# %%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)

# %% [markdown]
# ### XGBoost

# %%
xgb = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                      n_estimators=100, max_depth=3)
model = xgb.fit(x_train, y_train)
y_pred_test = xgb.predict_proba(x_test)[:, 1]
y_pred_train = xgb.predict_proba(x_train)[:, 1]
important_stats(y_train, y_pred_train, "train result summary: ")
important_stats(y_test, y_pred_test, "test result summary: ")


# %%
get_ipython().run_line_magic('matplotlib', 'inline')
plot_roc_curve(y_train, y_pred_train, y_test, y_pred_test)

# %% [markdown]
# ### GBDT

# %%
gbdt = GradientBoostingClassifier(random_state=0, n_estimators=10, max_depth=10)
gbdt = gbdt.fit(x_train, y_train)
y_pred_test = gbdt.predict_proba(x_test)[:, 1]
y_pred_train = gbdt.predict_proba(x_train)[:, 1]
important_stats(y_train, y_pred_train, "train result summary: ")
important_stats(y_test, y_pred_test, "test result summary: ")


# %%
plot_roc_curve(y_train, y_pred_train, y_test, y_pred_test)

# %% [markdown]
# ### LightGBM

# %%
gbm_clf = gbm.LGBMClassifier(
    boosting_type = 'gbdt',
    #num_leaves = ,
    #max_depth = ,
    learning_rate = 0.1
    #n_estimators = 
    #,subsample_for_bin =
    ,objective = 'binary'
    ,metric = 'binary_logloss'
    #,class_weight = 
    #,min_split_gain =
    #,min_split_weight =
    #,min_child_weight =
    #,min_child_samples =
    #,subsample =
    #,subsample_freq =
    #,colsample_bytree =
    ,reg_alpha = 5
    ,reg_lambda = 120
    ,importance_type = 'split' #will rank features by # of times it is used in model.'gain' for gain
    ,num_iterations = 1000
)


# %%
gbm_clf.fit(
    x_train, 
    y_train, 
    eval_metric = 'result', 
    verbose = True, 
    eval_set = [(x_val, y_val)],
    early_stopping_rounds = 20
)


# %%
y_pred_test = gbm_clf.predict_proba(x_test)[:, 1]
y_pred_train = gbm_clf.predict_proba(x_train)[:, 1]
important_stats(y_train, y_pred_train, "train result summary: ")
important_stats(y_test, y_pred_test, "test result summary: ")


# %%
plot_roc_curve(y_train, y_pred_train, y_test, y_pred_test)

# %% [markdown]
# ### GridSearch for GBDT

# %%
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                               min_samples_leaf=20,max_depth=8,max_features='sqrt',
                                                               subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(df.drop(columns = ['result','duration']),df['result'])
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# %%
param_test1 = {'n_estimators':range(80,151,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                               min_samples_leaf=20,max_depth=8,max_features='sqrt',
                                                               subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(df.drop(columns = ['result','duration']),df['result'])
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# %%
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(
    estimator = GradientBoostingClassifier(
        learning_rate=0.1, 
        n_estimators=120, 
        min_samples_leaf=20,
        max_features='sqrt', 
        subsample=0.8, 
        random_state=10
    ), 
    param_grid = param_test2, 
    scoring='roc_auc',
    iid=False, 
    cv=5)
gsearch2.fit(df.drop(columns = ['result','duration']),df['result'])
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


# %%
param_test3 = {'min_samples_leaf':range(60,101,10)}
gsearch3 = GridSearchCV(
    estimator = GradientBoostingClassifier(
        learning_rate=0.1, 
        n_estimators=120,
        max_depth=7,
        min_samples_split=700,
        max_features='sqrt', 
        subsample=0.8, 
        random_state=10
    ), 
    param_grid = param_test3, 
    scoring='roc_auc',
    iid=False, 
    verbose=1,
    cv=5
)

gsearch3.fit(df.drop(columns = ['result','duration']),df['result'])
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


# %%
gbdt_best = GradientBoostingClassifier(
    learning_rate=0.1, 
    n_estimators=100,
    max_depth=9, 
    min_samples_leaf =80, 
    min_samples_split =700, 
    max_features='sqrt', 
    subsample=0.8, 
    random_state=10
)
gbdt_best.fit(df.drop(columns = ['result','duration']),df['result'])


# %%
y_pred_test = gbdt_best.predict_proba(x_test)[:, 1]
y_pred_train = gbdt_best.predict_proba(x_train)[:, 1]
important_stats(y_train, y_pred_train, "train result summary: ")
important_stats(y_test, y_pred_test, "test result summary: ")


# %%
plot_roc_curve(y_train, y_pred_train, y_test, y_pred_test)

# %% [markdown]
# ### Features importance

# %%
gbdt_importance = gbdt_best.feature_importances_


# %%
from matplotlib.pyplot import figure
figure(num=None, figsize = (10,10))
indices = np.argsort(gbdt_importance)
plt.figure(1)
plt.title('GBDT Classifier Feature Importance')
plt.barh(range(len(indices)), gbdt_importance[indices], color = 'b', align = 'center')
gbdt_feat_names = x_train.columns
plt.yticks(range(len(indices)), gbdt_feat_names[indices])
plt.xlabel('Relative Importance: Gini')

# %% [markdown]
# ### Logistic Regression for predicting probabilities

# %%
lr = LogisticRegressionCV(solver = 'saga',
                           penalty = 'elasticnet',
                           l1_ratios = [0.1, 0.2, 0.3],
                           Cs = 20,
                           n_jobs = -1,
                           random_state = 0,
                           class_weight = 0.9
)
lr.fit(x_train,y_train)


# %%
y_pred_test = lr.predict_proba(x_test)[:, 1]
y_pred_train = lr.predict_proba(x_train)[:, 1]
important_stats(y_train, y_pred_train, "train result summary: ")
important_stats(y_test, y_pred_test, "test result summary: ")


# %%
plot_roc_curve(y_train, y_pred_train, y_test, y_pred_test)

