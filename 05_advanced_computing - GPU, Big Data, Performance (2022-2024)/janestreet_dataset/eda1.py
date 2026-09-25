#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, os, time

from sklearn.metrics import r2_score

import gc
# torch.cuda.empty_cache()
gc.collect()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

seed = 42
seed_everything(seed)

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping, LGBMRegressor as LGBMR
from IPython.display import display

#%%

df_tags = pd.read_csv('data/raw/features.csv')
display(df_tags)

print(df_tags.loc[df_tags.tag_2 == True].index)


#%%

df_resp = pd.read_csv('data/raw/responders.csv')
display(df_resp)

#%%
# Custom scoring method
def ScoreMetric(ypred,eval_data): #, weight):
    """
    This function is a modification of the ready-made R-square function with sample weight. 
    We have this as a column in the dataset
    """
    ytrue = eval_data.get_label()
    weight = np.array([1.749479]*len(ytrue))
    score = r2_score(ytrue, ypred, sample_weight = weight)
    return 'weighted_r2', score, True # Return: Name, score, higher is better

params = ({#"device"           : "gpu",
        "boosting_type"     : 'gbdt',
        "objective"        : "regression_l2",
        "metrics"          : "None",
        #"n_estimators"     : 1000,
        "max_depth"        : 10,
        "learning_rate"    : 0.03,
        "colsample_bytree" : 0.55,
        "subsample"        : 0.80,
        "random_state"     : seed,
        "reg_lambda"       : 1.25,
        "reg_alpha"        : 0.001,
        "verbose"          : -1,
        "n_jobs"           : 4,
         }
      )


#%%

df = pd.read_parquet('data/artifacts/train_symbol_0.parquet').reset_index(drop=True)

display(df.head(10))
print(df.shape)
print(df.isnull().sum().sum())

# %% Take a look at the target

df_1 = df.loc[df.date_id == 1]

df_1['responder_6'].plot()
plt.show()

#%% Keep only one dataset

no_days = len(df.date_id.unique())
daily_datapoints = []
for i in range(1,no_days+1):
    daily_datapoints.append(len(df.loc[df.date_id == i]))

plt.figure()
plt.hist(daily_datapoints)
plt.show()

display(df.head())

# %% Make train/test split. Use date 1500 for validation

train = df.loc[df.date_id < 1500]
test = df.loc[df.date_id >= 1500]

ytrain = train.pop('responder_6')
ytest = test.pop('responder_6')

features = ['date_id', 'time_id', 'feature_00', 'feature_01', 'feature_02', 'feature_03', 'feature_04',
       'feature_05', 'feature_06', 'feature_07', 'feature_08', 'feature_09', 'feature_10',
       'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16',
       'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22',
       'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28',
       'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34',
       'feature_35', 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40',
       'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46',
       'feature_47', 'feature_48', 'feature_49', 'feature_50', 'feature_51', 'feature_52',
       'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57', 'feature_58',
       'feature_59', 'feature_60', 'feature_61', 'feature_62', 'feature_63', 'feature_64',
       'feature_65', 'feature_66', 'feature_67', 'feature_68', 'feature_69', 'feature_70',
       'feature_71', 'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76',
       'feature_77', 'feature_78']

Xtrain = train[features]
Xtest = test[features]

print(len(Xtrain), len(ytrain))
print(len(Xtest), len(ytest))

# %% Train model

# Custom scoring method
def ScoreMetric(ypred,eval_data): #, weight):
    """
    This function is a modification of the ready-made R-square function with sample weight. 
    We have this as a column in the dataset
    """
    ytrue = eval_data.get_label()
    weight = np.array([1.749479]*len(ytrue))
    score = r2_score(ytrue, ypred, sample_weight = weight)
    return 'weighted_r2', score, True # Return: Name, score, higher is better


#Train model
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping, LGBMRegressor as LGBMR

params = ({#"device"           : "gpu",
        "boosting_type"     : 'gbdt',
        "objective"        : "regression_l2",
        "metrics"          : "None",
        "n_estimators"     : 5000,
        "max_depth"        : 10,
        "learning_rate"    : 0.03,
        "colsample_bytree" : 0.55,
        "subsample"        : 0.80,
        "random_state"     : seed,
        "reg_lambda"       : 1.25,
        "reg_alpha"        : 0.001,
        "verbose"          : -1,
        "n_jobs"           : 4,
         }
      )

trn_set = lgb.Dataset(Xtrain, ytrain)
tst_set = [lgb.Dataset(Xtest, ytest)]

rgr = lgb.train(params, trn_set,
    valid_sets           = tst_set,
    feval               = ScoreMetric,
    callbacks          = [log_evaluation(100), early_stopping(100, verbose = False)],
)

# %% Make check prediction

y_pred = rgr.predict(Xtest)
print('R2 score: ', r2_score(ytest, y_pred))
display(y_pred)


xx = np.arange(100)
plt.figure()
plt.plot(xx, ytest[:100], label='True')
plt.plot(xx, y_pred[:100], label='Predict')
plt.show()


# %% Try doing time series ---------------------------------------------------------

def make_dataset(df, N_LAGS=1, date_split=1500, 
                 responder_lags=False,
                 drop_features=True):
    df_temp = df.copy(deep=True)
    def get_lags(df_lags, N_LAGS, responder_lags):
        """ Make N cols with lags
        lag all the responder one step
        """
        
        for i in range(N_LAGS):
            df_lags[f'r6_lag{i+1}'] = df_lags.loc[:,'responder_6'].shift(i+1)

        if responder_lags:
            responder_features = [(f'responder_{i}',i) for i in [3] ] # range(9) if i not in [6]]
            for name, i in responder_features:
                df_lags[f'responder_{i}_lag'] = df_lags.loc[:,name].shift(1)


        #df_lags.dropna(inplace=True)    
        return df_lags


    df_responder_lags = get_lags(df_temp, N_LAGS, responder_lags)

    if drop_features:
        responder_features = [f'responder_{i}' for i in range(9) if i not in [6]]
        drop_features = ['symbol_id', 'weight', 'time_id', #'date_id', #, 'r6_lag1',  
                        ] + responder_features
        df_responder_lags = df_responder_lags.drop(columns=drop_features)
    else:
        r6_lags = [f'r6_lag{i}' for i in range(1,N_LAGS)]
        use_features = ['responder_6', #'date_id', 'time_id', #
                        'feature_48', 'feature_06', 'feature_59',] + r6_lags #'responder_3_lag'
        df_responder_lags = df_responder_lags[use_features]

    Xtrain = df_responder_lags.loc[df.date_id < date_split]
    Xtest = df_responder_lags.loc[df.date_id >= date_split]
    ytrain = Xtrain.pop('responder_6')
    ytest = Xtest.pop('responder_6')

    print(Xtrain.columns)
    return Xtrain, Xtest, ytrain, ytest
 
df_t = df.copy(deep=True)   
Xtrain, Xtest, ytrain, ytest = make_dataset(df_t,N_LAGS=50, drop_features=True)

# %% Train model

trn_set = lgb.Dataset(Xtrain, ytrain)
tst_set = [lgb.Dataset(Xtest, ytest)]

rgr2 = lgb.train(params, trn_set,
    valid_sets           = tst_set,
    feval               = ScoreMetric,
    callbacks          = [log_evaluation(100), early_stopping(100, verbose = False)],
)
# %% Make check prediction

y_pred = rgr2.predict(Xtest)
print('R2 score: ', r2_score(ytest, y_pred))
display(y_pred)


xx = np.arange(100)
plt.figure()
plt.plot(xx, ytest[:100], label='True')
plt.plot(xx, y_pred[:100], label='Predict')
plt.show()

#%%

feat_imp = rgr2.feature_importance(importance_type='gain')

df_feat = pd.DataFrame(data=feat_imp, index=Xtrain.columns, columns=['importance'])

df_feat.sort_values(by=['importance'], ascending=False).iloc[:10].plot.barh()


#%%
import warnings
warnings.filterwarnings("ignore")

test_length = 10000
start_time = time.time()
last_data = len(Xtrain)
r_lags = [f'r6_lag{i}' for i in range(1,50)]
y_lags = Xtrain.loc[last_data-1,r_lags]
# print(y_lags)
# print(type(y_lags))
# y_lags = y_lags.shift(1)
#print(y_lags)

y_pred_list = []
for i in range(len(Xtest)):
    data = Xtest.iloc[i,:].copy()
    data.loc[r_lags] = y_lags    
    
    y_pred = rgr2.predict(data)[0].astype(np.float32)
    y_pred_list.append(y_pred)
    y_lags = y_lags.shift(1) 
    y_lags.iloc[0] = y_pred
    if i % 20000 == 0:
        print(f'reached {i}') 
    
end_time = time.time()
print(f'Total time: {end_time - start_time}')

pred_arr = np.array(y_pred_list)



#%%

r2 = r2_score(ytest, np.array(y_pred_list))
print(f'r2 score: {r2}')

#%%

xx = range(100)
plt.figure()
plt.plot(xx, ytest.iloc[:100])
plt.plot(xx, pred_arr[:100])
plt.show()

#%%


dummy_set = lgb.Dataset(Xtest,ytest)
print(ScoreMetric(np.array(y_pred_list),tst_set[0]))


#%%


print(y_pred_list[0:5])
print(np.array(y_pred_list)[0:5])
print(y_pred2[0:5])

# %% Train all responders, predict from all responders

df_all = df.copy(deep=True)

responder_features = [f'responder_{i}' for i in range(9)]
drop_features = ['symbol_id', 'weight', 'date_id', 'time_id'] #+ responder_features
feature_features = [f'feature_{i:02}' for i in range(79)]
df_all = df_all.drop(columns=drop_features)


Xtrain = df_all.loc[df.date_id < 1500]
Xtest = df_all.loc[df.date_id >= 1500]
ytrain = Xtrain['responder_0']
Xtrain = Xtrain.drop(columns=responder_features)
ytest = Xtest['responder_0']
Xtest = Xtest.drop(columns=responder_features)

# %%
trn_set = lgb.Dataset(Xtrain, ytrain)
tst_set = [lgb.Dataset(Xtest, ytest)]

rgr2 = lgb.train(params, trn_set,
    valid_sets           = tst_set,
    feval               = ScoreMetric,
    callbacks          = [log_evaluation(100), early_stopping(100, verbose = False)],
)
# %% Make check prediction

y_pred = rgr2.predict(Xtest)
print('R2 score: ', r2_score(ytest, y_pred))
display(y_pred)





# %%
