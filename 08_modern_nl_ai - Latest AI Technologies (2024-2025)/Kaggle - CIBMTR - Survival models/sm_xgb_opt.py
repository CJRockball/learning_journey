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

xgb_params = {'tree_method': 'hist', #'gpu_hist', #
              'device': 'cuda',
                'objective': 'reg:squarederror', #'reg:absoluteerror', #'reg:logistic', #  'reg:squaredlogerror', #
                'random_state': 42,
                'eval_metric': "mae",
                # Using Fisher grouping of categorical features. None=One hot
                #"max_cat_to_onehot" : None,
                
                "max_depth":3,  
                "colsample_bytree":0.5, 
                "subsample":0.8, 
                "learning_rate":0.2, 
                "min_child_weight":5
                }

FEATURES = nums + cats   
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy()


def xgb_pred(df_X, df_y, df_train, cats_opt, params):
    oof_xgb = np.zeros(len(df_train))
    for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    
        # Set up data
        Xtrain = df_X.loc[train_idx,:].copy()
        ytrain = df_y.loc[train_idx].copy()
        Xvalid = df_X.loc[valid_idx,:].copy()
        yvalid = df_y.loc[valid_idx].copy()

        # Create DMatrix - is optimized for both memory efficiency and training speed.
        train_matrix = xgboost.DMatrix(data = Xtrain, label = ytrain, enable_categorical=True)
        valid_matrix = xgboost.DMatrix(data = Xvalid, label = yvalid, enable_categorical=True)
        evals = [(train_matrix, "train"), (valid_matrix, "validation")]
        # Create & Train the model
        model = xgboost.train(params=params, dtrain = train_matrix, evals=evals,
                              num_boost_round=2000,
                             early_stopping_rounds= 25, verbose_eval=0)

        # Make prediction
        predicts = model.predict(xgboost.DMatrix(Xvalid, enable_categorical=True))
        oof_xgb[valid_idx] = predicts



    c_score = c_index(df_train, oof_xgb)
    return c_score
        
#%%

# Optimization, check random cross features
# Baseline
start_time = time.time()
set_params = xgb_params
df_X2 = df_X.loc[:, FEATURES].copy()
c_score = xgb_pred(df_X2, df_y, df_train, cats, set_params)
best_score = c_score
end_time = time.time()
print(f'oof score {best_score}')
print(f'Total time: {end_time - start_time}')


def add_cat_feature(df, new_feature):
    fname = "_".join(new_feature)
    # Make new feature
    feat_length = len(new_feature)
    if feat_length == 2:
        df[fname] = df[new_feature[0]].astype(str) + df[new_feature[1]].astype(str)
    elif feat_length == 3:
        df[fname] = df[new_feature[0]].astype(str) + df[new_feature[1]].astype(str) +\
                    df[new_feature[2]].astype(str)
    elif feat_length == 4:
        df[fname] = df[new_feature[0]].astype(str) + df[new_feature[1]].astype(str) +\
                        df[new_feature[2]].astype(str) + df[new_feature[3]].astype(str)
    
    # Ordinal encode new features
    train_categories = df[fname].value_counts().sort_values().index.get_level_values(0).to_list()
    train_dict = {cname:i for i, cname in enumerate(train_categories)}    
    df[fname] = df[fname].map(train_dict).astype(np.int16)
    
    return df


MORE = []
start_time = time.time()
cats_opt = cats
for k in range(100):
    if k % 5 == 0:
        print(f"Number of tests run: {k}, Number of features: {df_X2.shape[1]}")
        
    ct = np.random.choice([2, 3, 4], 1)[0]
    new_feature = list( np.random.choice(cats, ct, replace=False))
    fname = "_".join(new_feature)
    df_X2 = add_cat_feature(df_X2, new_feature)
    cats_opt = cats_opt + [fname]
    
    new_score = xgb_pred(df_X2, df_y, df_train, cats_opt, set_params)
    
    if new_score > best_score:
        MORE.append(new_feature)
        best_score = new_score
        print(f'{new_feature}, oof score {new_score:.5f}')   
    else:
        df_X2 = df_X2.drop(columns=[fname])
        cats_opt.remove(fname)

end_time = time.time()
print(f'Total time {(end_time-start_time)/60}')      
print("DONE")

# %%

print(MORE)

#%%
df_train2 = df_train.drop(columns=target+['ID']).copy(deep=True)

add_list = [['race_group', 'tbi_status'], ['arrhythmia', 'prior_tumor', 'diabetes'], ['cardiac', 'tce_div_match', 'pulm_severe'], ['tce_match', 'cyto_score_detail'], ['hepatic_severe', 'tce_match', 'ethnicity'], ['melphalan_dose', 'gvhd_proph', 'prod_type'], ['sex_match', 'cyto_score_detail']]
add_col = []

def make_cats_features(df_train, df_test, add_list):
    for lst in add_list:
        fname = "_".join(lst)
        if len(lst) == 2:
            df_train[fname] = df_train[lst[0]].astype(str) + df_train[lst[1]].astype(str)
            df_test[fname] = df_test[lst[0]].astype(str) + df_test[lst[1]].astype(str)
        elif len(lst) == 3:
            df_train[fname] = df_train[lst[0]].astype(str) + df_train[lst[1]].astype(str) + df_train[lst[2]].astype(str)
            df_test[fname] = df_test[lst[0]].astype(str) + df_test[lst[1]].astype(str) + df_test[lst[2]].astype(str)
        elif len(lst) == 4:
            df_train[fname] = df_train[lst[0]].astype(str) + df_train[lst[1]].astype(str) + df_train[lst[2]].astype(str) + df_train[lst[3]].astype(str)
            df_test[fname] = df_test[lst[0]].astype(str) + df_test[lst[1]].astype(str) + df_test[lst[2]].astype(str) + df_test[lst[3]].astype(str)       
        add_col.append(fname)
        
    return df_train, df_test, add_col            

df_train2, df_test2, add_col = make_cats_features(df_train2, df_test, add_list)
cats = cats + add_col

def ord_enc(df, cats_mod):
    for cname in cats_mod:
        train_categories = df[cname].value_counts().sort_values().index.get_level_values(0).to_list()
        train_dict = {fname:i for i, fname in enumerate(train_categories)} 
        df[cname] = df[cname].map(train_dict).astype(np.int16)
    return df

df_train2 = ord_enc(df_train2, add_col)

# Set up dataframes
print('set up dfs')
df_X3 = df_train2.copy()
X_train, X_valid, y_train, y_valid = train_test_split(df_X3, df_y, test_size=0.2, random_state=45)


# Create DMatrix - is optimized for both memory efficiency and training speed.
train_matrix = xgboost.DMatrix(data = X_train, label = y_train, enable_categorical=True)
valid_matrix = xgboost.DMatrix(data = X_valid, label = y_valid, enable_categorical=True)
evals = [(train_matrix, "train"), (valid_matrix, "validation")]
# Create & Train the model
model = xgboost.train(params=xgb_params, dtrain = train_matrix, evals=evals,
                              num_boost_round=2000,
                             early_stopping_rounds= 25, verbose_eval=100)
y_valid_pred = model.predict(xgboost.DMatrix(X_valid, enable_categorical=True))

idx = X_valid.index.to_list()
c_score = c_index(df_train.iloc[idx], y_valid_pred)
print(c_score)

# %%

imp_data = model.get_score(importance_type='total_cover')

df_imp = pd.DataFrame(list(imp_data.values()), index=list(imp_data.keys()), columns=['cover'])

plt.figure()
df_imp.sort_values(by=['cover'], ascending=False).iloc[:15,:]['cover'].plot(kind='bar')
plt.show()

# %%
