#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
import lightgbm as lgb

from scipy.stats import rankdata 
import pandas.api.types
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
   
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

#GET SCORE
def c_index(df_train, data_array):
    y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df_train[["ID"]].copy()
    y_pred["prediction"] = data_array
    m = score(y_true.copy(), y_pred.copy(), "ID")
    return m

#%% Get data
print('load')
df_train = pd.read_csv('data/ra/train.csv')
df_test = pd.read_csv('data/ra//test.csv')

# Define categories
target =['efs', 'efs_time']
# nums = ['donor_age', 'age_at_hct'] #, 'efs_time']
# cats = [name for name in df_train.columns if name not in (nums + target + ['ID'])]

cats = df_train.select_dtypes(include=["object_", 'bool']).columns
cats = [cat for cat in cats if cat not in target+['ID']]
nums = df_train.select_dtypes(include=['number']).columns
nums = [num for num in nums if num not in target+['ID']]

# encode cats
for cname in cats:
    df_train[cname] = df_train[cname].fillna('Missing')
    df_test[cname] = df_test[cname].fillna('Missing')

    train_categories = df_train[cname].value_counts().sort_values().index.get_level_values(0).to_list()
    train_dict = {fname:i for i, fname in enumerate(train_categories)}

    test_categories = df_test[cname].unique()
    test_dict = {}
    for fname in test_categories:
        if fname in train_categories:
            test_dict[fname] = train_dict[fname]
        elif df_train[cname].isnull().any(): 
            test_dict[fname] = train_dict['Missing']
        else:
            test_dict[fname] = 0    
    
    df_train[cname] = df_train[cname].map(train_dict).astype(np.int16)
    df_test[cname] = df_test[cname].map(test_dict).astype(np.int16)       

# Encode num
print('nums')
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


# Set up dataframes
print('set up dfs')
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target).copy()
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')

FEATURES = nums + cats

#%%

# KM
KFOLD = 5
oof_pred = np.zeros(len(df_target))
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

for train_index, valid_index in kf.split(df_target):
    train_data = df_target.iloc[train_index]
    valid_data = df_target.iloc[valid_index]
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
    oof_pred[valid_index] = kmf.survival_function_at_times(valid_data['efs_time']).values

df_target['km'] = oof_pred
m = c_index(df_train, df_target['km'])
print('km', m)
df_train_TE = df_train.copy(deep=True)

#%%
from sklearn.preprocessing import TargetEncoder


def cv_func(df_X, df_y, df_train, cats_opt, kf, train_fcn, params, NUM_ROUNDS):
    oof_lgb = np.zeros(len(df_train))
    full_index = []
    add_names = []
    for cat in cats:
        add_names.append(str(cat) + '_TE')
        
    for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    
    # Set up data
        Xtrain = df_X.loc[train_idx,:].copy()
        ytrain = df_y.loc[train_idx].copy()
        Xvalid = df_X.loc[valid_idx,:].copy()
        yvalid = df_y.loc[valid_idx].copy()
        full_index = full_index + list(valid_idx)
        
        Xtrain_TE = Xtrain.copy()
        Xvalid_TE = Xvalid.copy()
        enc_auto = TargetEncoder(smooth="auto", random_state=42)        
        Xtrain_TE[add_names] = enc_auto.fit_transform(Xtrain_TE[cats], ytrain.values.ravel())
        Xvalid_TE[add_names] = enc_auto.transform(Xvalid_TE[cats])
        
        y_pred = train_fcn(Xtrain_TE, Xvalid_TE, ytrain, yvalid, cats_opt, params, NUM_ROUNDS)

        oof_lgb[valid_idx] = y_pred

    full_score = c_index(df_train, oof_lgb)
    return full_score


def xgb_train(Xtrain, Xvalid, ytrain, yvalid, cats_opt, params, NUM_ROUNDS):

    # Create DMatrix - is optimized for both memory efficiency and training speed.
    train_matrix = xgboost.DMatrix(data = Xtrain, label = ytrain, enable_categorical=True)
    valid_matrix = xgboost.DMatrix(data = Xvalid, label = yvalid, enable_categorical=True)
    evals = [(train_matrix, "train"), (valid_matrix, "validation")]
    # Create & Train the model
    model = xgboost.train(params=params, dtrain = train_matrix, evals=evals,
                              num_boost_round=2000,
                             early_stopping_rounds= 25, verbose_eval=100)

    # Make prediction
    y_pred = model.predict(xgboost.DMatrix(Xvalid, enable_categorical=True))
    
    return y_pred


#%%

xgb_params = {'tree_method': 'hist', #'gpu_hist', #
              'device': 'cuda',
                'objective': 'reg:squarederror', #'reg:absoluteerror', #'reg:logistic', #  'reg:squaredlogerror', #
                'random_state': 42,
                'eval_metric': "rmse",
                # Using Fisher grouping of categorical features. None=One hot
                #"max_cat_to_onehot" : None,
                
                "max_depth":3,  
                "colsample_bytree":0.5, 
                "subsample":0.8, 
                "learning_rate":0.2, 
                "min_child_weight":5
                }

NUM_ROUNDS = 2000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy()
#df_X = df_train_TE.drop(columns=target).copy()

start_time = time.time()
full_score = cv_func(df_X, df_y, df_train, cats, kf, xgb_train, xgb_params, NUM_ROUNDS)
print(full_score)
end_time = time.time()
print(f'Total time {end_time - start_time}')

#%%
