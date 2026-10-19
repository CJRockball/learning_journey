#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
#from catboost import CatBoostRegressor, Pool

"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.
 
It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.
 
The concordance index is a value between 0 and 1 where:
 
0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas.api.types
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
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

#%% Get data

df_train = pd.read_csv('data/ra/train.csv')
df_train = df_train.drop(columns=['ID'])
df_train = df_train.astype({'efs':bool})
display(df_train)
print(df_train.shape)

df_test = pd.read_csv('data/ra/test.csv')
df_test = df_test.drop(columns=['ID'])
print(df_test.shape)
display(df_test)

# %%
cats_count = {2:0, 50:0, 'many':0}

for name in df_train.columns:
    num_cats = df_train[name].nunique()
    if num_cats == 2:
        cats_count[2] += 1    
    elif (num_cats < 50 and num_cats > 2):
        cats_count[50] += 1
    else: 
        cats_count['many'] += 1
        print(name)

print(cats_count)

# %% Define categories

target =['efs', 'efs_time']
nums = ['donor_age', 'age_at_hct'] #, 'efs_time']
cats = [name for name in df_train.columns if name not in (nums + target)]

#%% encode cats


for cname in cats:
    print(cname)    
    
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

# %% Feature Engineering

# # Fill na for nums. Put mean.    
# add_cols = []
# for cname in nums:
#     print(f'fillna for {cname}')
#     # Mark all cols ith nans
#     if df_train[cname].isnull().values.any():
#         print('Add nan col')
#         df_train[f"is_{cname}_na"] = df_train[cname].isna().astype(int)
#         df_test[f"is_{cname}_na"] = df_test[cname].isna().astype(int)
#         add_cols.append(f"is_{cname}_na")

#     # Fill all nans ith mean
#     m = df_train[cname].mean()
#     df_train[cname] = df_train[cname].fillna(m)
#     df_test[cname]  = df_test[cname].fillna(m)

# cats = cats + add_cols

for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)

# %%

kmf = KaplanMeierFitter()
kmf.fit(df_train['efs_time'].values, df_train['efs'].values)

#%% Plot survival function
from lifelines.utils import median_survival_times

#kmf.survival_function_.plot()
kmf.plot_survival_function()
plt.title('Built in survival function kmf')
mean_surv = kmf.median_survival_time_
median_ci = median_survival_times(kmf.confidence_interval_)

print(f'Mean Survival time: {mean_surv}')
print(median_ci)

#%%

survival_probabilities = kmf.survival_function_at_times(df_train['efs_time']).values.flatten()

print(type(survival_probabilities))
print(survival_probabilities.shape)
print(df_train.shape)
print(survival_probabilities[:20])

#%% Plot survivial function by race

races = df_train['race_group'].unique()

for i, race in enumerate(races):
    ax = plt.subplot(2, 3, i+1)
    
    ix = df_train['race_group'] == race
    kmf.fit(df_train.loc[ix,'efs_time'], df_train.loc[ix,'efs'], label=race)

    kmf.plot_survival_function(ax=ax, legend=False)
    
    plt.title(race)
    plt.xlim(0,20)
plt.tight_layout()


plt.figure()
for race in races:
    ix = df_train['race_group'] == race    
    print(f"{race}: {ix.sum()}")
    kmf.fit(df_train.loc[ix,'efs_time'], df_train.loc[ix,'efs'], label=race)
    data = kmf.survival_function_
    plt.plot(data.index, data.iloc[:,0])

plt.legend()
plt.show()
    
#%% Split data

train_l = int(0.8*len(df_train))
Xtrain = df_train.iloc[:train_l,:]
Xvalid = df_train.iloc[train_l:,:]

#%%
print(Xtrain.shape)
print(Xvalid.shape)
print(train_l)

# %% Try fitting the function and plotting
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(Xtrain, duration_col='efs_time', event_col='efs', strata='race_group')

cph.print_summary()

# %% 

cox_ph_survival_funcs = cph.predict_survival_function(Xvalid)
cox_pred = np.transpose(cox_ph_survival_funcs)

xx = range(cox_pred.shape[1])
plt.figure()
for i in range(100):
    plt.plot(xx, cox_pred.iloc[i,:], label=i)
#plt.legend()
plt.show()

#%%

med_life_exp = cph.predict_median(Xvalid) 
print(med_life_exp)
print(type(med_life_exp))

exp_life = cph.predict_expectation(Xvalid)

#%% Get one value per ro for ranking

# Predict partial hazards on the validation set
val_pred = cph.predict_partial_hazard(Xvalid)

print(val_pred.shape)

# Calculate the concordance index
c_index = concordance_index(Xvalid['efs_time'], -val_pred, Xvalid['efs'])
print(f"Concordance Index on Validation Set: {c_index:.4f}")

#%% Repeat ith sklearn survival
from sksurv.nonparametric import kaplan_meier_estimator
import seaborn as sns; sns.set_style("darkgrid")

Xtrain['efs'] = Xtrain['efs'].astype('boolean')
Xvalid['efs'] = Xvalid['efs'].astype('boolean')

# Get pred-times and probabilities, ordered
times, km_survival_probabilities = kaplan_meier_estimator(
    Xtrain["efs"], Xtrain["efs_time"])

# Plot data
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(times, km_survival_probabilities, label="Kaplan-Meier")
ax.set(title="Mean survival curve", xlabel="Time (days)")
ax.legend();


# %% CHeck
# time and km_surv are ordered numpy arrays.
# ith a length of 16337 datapoints because that's the number of (1 in efs)
print(times)
print(len(times))
print(km_survival_probabilities)
print(Xtrain.efs_time.max())
print(len(Xtrain))
print(len(Xtrain.loc[Xtrain['efs'] == 1]))

# %%
from scipy.interpolate import interp1d
n_steps=300

km_predict = interp1d(
    times,
    km_survival_probabilities,
    kind="previous",
    bounds_error=False,
    fill_value="extrapolate",
)

# Make an evenly space grid of points from min to max. 
# Trim 0.5% for end prediction problems
def make_test_time_grid(Xtrain, n_steps=300):
    """Bound times to the range of duration."""
    # Some survival models can fail to predict near the boundary of the
    # range of durations observed on the training set.
    observed_duration = Xtrain.loc[Xtrain.efs == True,["efs_time"]].values
    
    # trim 1% of the span, 0.5% on each end:
    span = observed_duration.max() - observed_duration.min()
    start = observed_duration.min() + 0.005 * span
    stop = observed_duration.max() - 0.005 * span
    return np.linspace(start, stop, num=n_steps)


time_grid = make_test_time_grid(Xtrain)


print(time_grid)
print(time_grid.shape)
print(Xtrain.efs_time.max())
print(Xtrain.loc[Xtrain.efs == True].efs_time.max())

# %%
# KM predicts one unconditional curve for all rows i.e. doesn't depend on features
# calculate km_curve from the time_grid
Xvalid_filter = Xvalid.loc[Xvalid.efs_time < 120,:]

km_curve = km_predict(time_grid)
y_pred_km_test = np.vstack([km_curve] * Xvalid_filter.shape[0])

# Xticks are rows, not days
plt.figure()
plt.plot(range(300), km_curve)
plt.show()


# %%
from sksurv.metrics import brier_score

# Get ytrain and ytest in the right format, numpy recarray
def as_sksurv_recarray(y_frame):
    """Return scikit-survival's specific target format."""
    y_recarray = np.empty(
        shape=y_frame.shape[0],
        dtype=[("efs", np.bool_), ("efs_time", np.float64)],
    )
    y_recarray["efs"] = y_frame["efs"]
    y_recarray["efs_time"] = y_frame["efs_time"]
    return y_recarray


_, km_brier_scores = brier_score(
    survival_train=as_sksurv_recarray(Xtrain[['efs', 'efs_time']]),
    survival_test=as_sksurv_recarray(Xvalid_filter[['efs', 'efs_time']]),
    estimate=y_pred_km_test,
    times=time_grid,
)

# %%

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_grid, km_brier_scores, label="KM on test data");
ax.set(
    title="Time-varying Brier score of Kaplan Meier estimation (lower is better)",
    xlabel = "time",
)
ax.legend();


#%% Integrated Briar score for comparison
from sksurv.metrics import integrated_brier_score

km_ibs_test = integrated_brier_score(
    survival_train=as_sksurv_recarray(Xtrain[['efs', 'efs_time']]),
    survival_test=as_sksurv_recarray(Xvalid_filter[['efs', 'efs_time']]),
    estimate=y_pred_km_test,
    times=time_grid,
)

print(f"IBS of Kaplan-Meier estimator on test set: {km_ibs_test:.3f}")

# %% Use a straight line as survival function

linear_survival_curve = np.linspace(1.0, 0.0, time_grid.shape[0])
constant_linear_survival_curves = [linear_survival_curve] * Xvalid_filter.shape[0]

linear_survival_ibs_test = integrated_brier_score(
    survival_train=as_sksurv_recarray(Xtrain[['efs', 'efs_time']]),
    survival_test=as_sksurv_recarray(Xvalid_filter[['efs', 'efs_time']]),
    estimate=constant_linear_survival_curves,
    times=time_grid,
)
print(f"IBS of linear survival estimator on test set: {linear_survival_ibs_test:.3f}")
# Clearly orse
# %% c-index

from sksurv.metrics import concordance_index_censored


def compute_c_index(event, duration, survival_curves):
    survival_curves = np.asarray(survival_curves)
    if survival_curves.ndim != 2:
        raise ValueError(
            "`survival_probs` must be a 2d array of "
            f"shape (n_samples, times), got {survival_curves.shape}"
        )
    assert event.shape[0] == duration.shape[0], survival_curves.shape[0]
    
    # Cumulative hazard is also known as risk.
    cumulative_hazard = survival_to_risk_estimate(survival_curves)
    metrics = concordance_index_censored(event, duration, cumulative_hazard)
    return metrics[0]


def survival_to_risk_estimate(survival_probs_matrix):
    return -np.log(survival_probs_matrix + 1e-8).sum(axis=1)

Xvalid_filter['efs'] = Xvalid_filter['efs'].astype(np.bool_)
km_c_index_test = compute_c_index(Xvalid_filter["efs"], Xvalid_filter["efs_time"], y_pred_km_test)
print(km_c_index_test)
# 0.5 is equivalent to guessing. Because KM makes a generic model and doesn't include covariates
# Make sure in dtype efs is np.bool_

# %% Make ohe for Cox prop hazard model

c_check = [cat for cat in cats if cat not in ['ID']]
df_train1 = pd.get_dummies(df_train, columns=c_check, drop_first=True)
    
display(df_train1)

# %%

train_l = int(0.8*len(df_train1))
Xtrain1 = df_train1.iloc[:train_l,:]
Xvalid1 = df_train1.iloc[train_l:,:]
Xvalid2 = Xvalid1.loc[Xvalid1['efs_time'] < 120]

ytrain1 = Xtrain1[['efs', 'efs_time']]
ytrain1 = ytrain1.astype({"efs": int})
Xtrain1 = Xtrain1.drop(columns=['efs', 'efs_time'])

yvalid1 = Xvalid1[['efs', 'efs_time']]
Xvalid1 = Xvalid1.drop(columns=['efs', 'efs_time'])

yvalid2 = Xvalid2[['efs', 'efs_time']]
Xvalid2 = Xvalid2.drop(columns=['efs', 'efs_time'])

#%%

print(df_train.columns)

#%%
import time
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sksurv.linear_model import CoxPHSurvivalAnalysis

start_time = time.time()
cox_ph = CoxPHSurvivalAnalysis(alpha=1e-4)
cox_ph.fit(Xtrain1.values, as_sksurv_recarray(ytrain1))

cox_ph_survival_funcs = cox_ph.predict_survival_function(Xvalid2.values)

end_time = time.time()
print(f'Total time: {end_time - start_time}')

#%%

fig, ax = plt.subplots()
for idx, cox_ph_survival_func in enumerate(cox_ph_survival_funcs[:10]):
    survival_curve = cox_ph_survival_func(time_grid)
    ax.plot(time_grid, survival_curve, label=idx)
ax.set(
    title="Survival probabilities $\hat{S(t)}$ of Cox PH",
    xlabel="time (days)",
    ylabel="S(t)",
)
plt.legend();

#%%




#%%

class SurvivalAnalysisEvaluator:
    
    def __init__(self, y_train, y_test, time_grid):
        self.model_data = {}
        self.y_train = as_sksurv_recarray(y_train)
        self.y_test = as_sksurv_recarray(y_test)
        self.time_grid = time_grid
        
    def add_model(self, model_name, survival_curves):
        survival_curves = np.asarray(survival_curves)
        _, brier_scores = brier_score(
            survival_train= self.y_train,
            survival_test= self.y_test,
            estimate=survival_curves,
            times=self.time_grid,
        )
        ibs = integrated_brier_score(
            survival_train=self.y_train,
            survival_test= self.y_test,
            estimate=survival_curves,
            times=self.time_grid,
        )
        c_index = compute_c_index(
            self.y_test["efs"],
            self.y_test["efs_time"],
            survival_curves,
        )
        self.model_data[model_name] = {
            "brier_scores": brier_scores,
            "ibs": ibs,
            "c_index": c_index,
            "survival_curves": survival_curves,
        }

    # Get ytrain and ytest in the right format, numpy recarray
    def as_sksurv_recarray(y_frame):
        """Return scikit-survival's specific target format."""
        y_recarray = np.empty(
        shape=y_frame.shape[0],
        dtype=[("efs", np.bool_), ("efs_time", np.float64)],)
        y_recarray["efs"] = y_frame["efs"]
        y_recarray["efs_time"] = y_frame["efs_time"]
        return y_recarray

    def metrics_table(self):
        return pd.DataFrame([
            {
                "Model": model_name,
                "IBS": info["ibs"],
                "C-index": info["c_index"],
            }
            for model_name, info in self.model_data.items()
        ]).round(decimals=4)
        
    def plot(self, model_names=None):
        if model_names is None:
            model_names = list(self.model_data.keys())
        fig, ax = plt.subplots(figsize=(12, 5))
        self._plot_brier_scores(model_names, ax=ax)

    def _plot_brier_scores(self, model_names, ax):
        for model_name in model_names:
            info = self.model_data[model_name]
            ax.plot(
                self.time_grid,
                info["brier_scores"],
                label=f"{model_name}, IBS:{info['ibs']:.3f}");
        ax.set(
            title="Time-varying Brier score (lower is better)",
            xlabel="time (days)",
        )
        ax.legend()
        
    def __call__(self, model_name, survival_curves, model_names=None):
        self.add_model(model_name, survival_curves)
        self.plot(model_names=model_names)
        return self.metrics_table()

evaluator = SurvivalAnalysisEvaluator(ytrain1, yvalid2, time_grid)
evaluator.add_model("Kaplan-Meier", y_pred_km_test)
evaluator.add_model("Constant linear", constant_linear_survival_curves)
evaluator.plot()
evaluator.metrics_table()

# %%

cox_survival_curves = np.vstack(
    [
        cox_ph_survival_func(time_grid)
        for cox_ph_survival_func in cox_ph_survival_funcs
    ]
)
evaluator("Cox PH", cox_survival_curves)



# %%


from sklearn.preprocessing import SplineTransformer
from sklearn.kernel_approximation import Nystroem


spline_preprocessor = make_column_transformer(
    (SplineTransformer(), nums),
    verbose_feature_names_out=False,
)
poly_cox_ph = make_pipeline(
    spline_preprocessor,
    Nystroem(kernel="poly", degree=2, n_components=300),
    CoxPHSurvivalAnalysis(alpha=1e-2)
)
poly_cox_ph.fit(Xtrain1, as_sksurv_recarray(ytrain1))
poly_cox_ph_survival_funcs = poly_cox_ph.predict_survival_function(Xvalid2)


poly_cox_ph_survival_curves = np.vstack(
    [
        poly_cox_ph_survival_func(time_grid)
        for poly_cox_ph_survival_func in poly_cox_ph_survival_funcs
    ]
)
evaluator("Polynomial Cox PH", poly_cox_ph_survival_curves)

#%% Put y in right format. structured array

ytrain1_list = [(a,b) for a,b in zip(ytrain1['efs'].tolist(), ytrain1['efs_time'].tolist()) ]
ytrain1_arr = np.array(ytrain1_list, dtype=[('cens', '?'), ('time', '<f8')] )
#print(ytrain_arr)


# %%
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(n_estimators=10, max_depth=8, n_jobs=-1)

# Make sure efs is np.bool_
rsf.fit(Xtrain1, as_sksurv_recarray(ytrain1))


# %%
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

gbsa = GradientBoostingSurvivalAnalysis()
# Make sure efs is np.bool_
gbsa.fit(Xtrain1, as_sksurv_recarray(ytrain1))

# %% Boosting fit

gbsa_survival_funcs = gbsa.predict_survival_function(Xvalid1)

fig, ax = plt.subplots()
for idx, survival_func in enumerate(gbsa_survival_funcs[:5]):
    survival_curve = survival_func(time_grid)
    ax.plot(time_grid, survival_curve, label=idx)
ax.set(
    title="Survival probabilities $\hat{S}(t)$ of Random Survival Forest",
    xlabel="time (days)",
    ylabel="S(t)",
)
plt.legend();


#%% Home made surivival and hazard functions

def survival_function(df):
    survival_df = df[['efs', 'efs_time']].groupby('efs_time').agg(['size', 'sum']).droplevel(0,axis=1).astype(int)
    survival_df['n_at_risk'] = survival_df['size'].sum() - survival_df['size'].shift().fillna(0).cumsum().astype(int)
    hazard = survival_df['sum'] / survival_df['n_at_risk']
    survival_df['cumulative_hazard'] = np.cumsum(hazard) # nelson-aalen_estimator
    survival_df['survival_probability'] = (1-hazard).cumprod() #kaplan-meier_estimator

    return survival_df

#%%n plot
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
race_groups = np.unique(df_train.race_group)
plt.figure(figsize=(6, 8))

plt.subplot(2, 1, 1)
survival_df = survival_function(df_train)
plt.step(survival_df.index, survival_df['survival_probability'], c='k', where="post", label='[Overall]')
plt.xlabel('efs_time')
for race_group in race_groups:
    subset = df_train.query('race_group == @race_group')
    survival_df = survival_function(subset)
    plt.step(survival_df.index, survival_df['survival_probability'], where="post", label=race_group)
plt.xlabel('efs_time')
plt.legend(loc='upper right')
plt.title('Survival function (Kaplan–Meier) by race group')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0)) # percent of xmax

plt.subplot(2, 1, 2)
survival_df = survival_function(df_train)
plt.step(survival_df.index, survival_df['cumulative_hazard'], c='k', where="post", label='[Overall]')
plt.xlabel('efs_time')
for race_group in race_groups:
    subset = df_train.query('race_group == @race_group')
    survival_df = survival_function(subset)
    plt.step(survival_df.index, survival_df['cumulative_hazard'], where="post", label=race_group)
plt.xlabel('efs_time')
plt.legend(loc='lower right')
plt.title('Cumulative hazard (Nelson–Aalen) by race group')

plt.tight_layout()
plt.show()

#%%


def evaluate_fold(y_va_pred, fold):
    """Compute and print the metrics (concordance index) per race group for a single fold.

    Global variables:
    - train, X_va, idx_va
    - The metrics are saved in the global list all_scores.
    """
    metric_list = []
    for race in race_groups:
        mask = X_va.race_group.values == race
        c_index_race = concordance_index(
            df_train.efs_time.iloc[idx_va][mask],
            - y_va_pred[mask],
            df_train.efs.iloc[idx_va][mask]
        )
        # print(f"# {race:42} {c_index_race:.3f}")
        metric_list.append(c_index_race)
    fold_score = np.mean(metric_list) - np.sqrt(np.var(metric_list))
    print(f"# Total fold {fold}:{' ':29} {fold_score:.3f} mean={np.mean(metric_list):.3f} std={np.std(metric_list):.3f}")
    all_scores.append(metric_list)

def display_overall(label):
    """Compute and print the overall metrics (concordance index)"""
    df = pd.DataFrame(all_scores, columns=race_groups)
    df['mean'] = df[race_groups].mean(axis=1)
    df['std'] = np.std(df[race_groups], axis=1)
    df['score'] = df['mean'] - df['std']
    df = df.T
    df['Overall'] = df.mean(axis=1)
    temp = df.drop(index=['std']).values
    print(f"# Overall:                                   {df.loc['score', 'Overall']:.3f} {label}")
    # all_model_scores[label] = df.loc['score', 'Overall']
    display(df
            .iloc[:len(race_groups)]
            .style
            .format(precision=3)
            .background_gradient(axis=None, vmin=temp.min(), vmax=temp.max(), cmap="cool")
            .concat(df.iloc[len(race_groups):].style.format(precision=3))
           )

#%% SPECIAL XGBOOST library
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(shuffle=True, random_state=1)
TARGET = ['efs', 'efs_time']
FEATURES = [col for col in df_train.columns if col not in TARGET]
# XGBoost Cox regression
y = np.where(df_train.efs == 1, df_train.efs_time, -df_train.efs_time)
all_scores = []
for fold, (idx_tr, idx_va) in enumerate(kf.split(df_train, df_train.race_group)):
    X_tr = df_train.iloc[idx_tr][FEATURES]
    X_va = df_train.iloc[idx_va][FEATURES]
    y_tr = y[idx_tr]
    
    xgb_cox_params = {'objective': 'survival:cox', 'grow_policy': 'depthwise', 
                      'n_estimators': 700, 'learning_rate': 0.0254, 'max_depth': 8, 
                      'reg_lambda': 0.116, 'reg_alpha': 0.139, 'min_child_weight': 23.8,
                      'colsample_bytree': 0.59, 'subsample': 0.7, 'tree_method': 'hist',
                      'enable_categorical': True}
    model = xgboost.XGBRegressor(**xgb_cox_params)
    model.fit(X_tr, y_tr) # negative values are considered right censored
    y_va_pred = model.predict(X_va) # predicts hazard factor
    evaluate_fold(y_va_pred, fold)
display_overall('Cox Proportional Hazards XGBoost')

#%%

print(df_train.columns)


# %% CV cox's ph

#df_y = df_train1[['efs', 'efs_time']]
#df_X = df_train1.drop(columns=['efs', 'efs_time'])
all_scores = []
for fold, (idx_tr, idx_va) in enumerate(kf.split(df_train1, df_train1.race_group)):
    Xtrain = df_train1.iloc[idx_tr,:].copy()
#    ytrain = df_y.iloc[idx_tr].copy()
    Xvalid = df_train1.iloc[idx_va,:].copy()
#    yvalid = df_y.iloc[idx_va,:].copy()

    model = CoxPHFitter(penalizer=.01)
    model.fit(Xtrain, duration_col='efs_time', event_col='efs')
    # model.print_summary()
    yvalid_pred = model.predict_partial_hazard(Xvalid[FEATURES])
    Xvalid['race_group'] = df_train1.race_group.iloc[idx_va]
    evaluate_fold(yvalid_pred, fold)
    
display_overall('Cox Proportional Hazards Linear')


