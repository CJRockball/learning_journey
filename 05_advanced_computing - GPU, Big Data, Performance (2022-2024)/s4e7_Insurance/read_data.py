#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/home/patrick/Python/timeseries/weather/kaggle/Insurance_s4e7')

print(os.getcwd())

# %%


def prep_data(load_path, save_path, train=True):
    df = pd.read_csv(load_path) 
    print(df.info())

    # Data prep
    # Basic data mod of train data
    df = df.drop(columns=['id'])
    df['Gender'] = df.Gender.replace({"Female":0, 'Male':1}).astype(np.int8)
    df['Vehicle_Age'] = df.Vehicle_Age.replace({'< 1 Year':0, '1-2 Year':1, '> 2 Years':2}).astype(np.int8)
    df['Vehicle_Damage']  = df.Vehicle_Damage.replace({'No':0, 'Yes':1}).astype(np.int8)

    list_int = ['Vehicle_Age','Vintage','Policy_Sales_Channel','Region_Code']
    list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
    list_float32 = ['Annual_Premium','Age']
    list_target = ['Response']

    df[list_int] = df[list_int].astype(np.int32)
    # Optimize integer columns
    if train:
        int_columns = list_int + list_bool + list_target #df.select_dtypes(include=['int64']).columns
    else:
        int_columns = list_int + list_bool
        
    for col in int_columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype(np.int8)
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype(np.int16)
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype(np.int32)

    df[list_float32] = df[list_float32].astype(np.float32)

    #df = pd.get_dummies(df, columns=['Vehicle_Age'])
    print(df.info())
    df.to_parquet(save_path)
    return 

#prep_data('data/raw/train.csv', 'data/artifacts/train.parquet')
# 1+ GB -> 209 MB
prep_data('data/raw/test.csv', 'data/artifacts/test.parquet', train=False)
# 644 MB -> 132 MB

#%% Normalize test data
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

list_int32 = ['Vehicle_Age','Vintage','Policy_Sales_Channel','Region_Code']
list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
list_float32 = ['Annual_Premium','Age']
list_category = ['Vehicle_Age', 'Vintage','Policy_Sales_Channel','Region_Code']
#list_target = ['Response']

df = pd.read_parquet('data/train_proc.parquet')

df.drop(columns=['Response'], inplace=True)
df[list_bool] = df[list_bool].astype(np.int8)

scaler = StandardScaler()
df[list_float32] = scaler.fit(df[list_float32])

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
df[list_int32] = oe.fit_transform(df[list_int32])

del df

# Read test data
df_test = pd.read_parquet('data/df_test.parquet')
# Change bool type to int
df_test[list_bool] = df_test[list_bool].astype(np.int8)
# Normalize and standardize contisuous data
df_test[list_float32] = scaler.transform(df_test[list_float32])
# Recode ordinal data
df_test[list_int32] = oe.transform(df_test[list_int32])
# Put all unseen cats in 0, Change list_int32 type to typw 32
df_test[list_int32] = df_test[list_int32].fillna(0).astype(np.int32)

df_test.to_parquet('data/df_test_norm.parquet')

display(df_test.head())
print(df_test.info())

#%% Feature selection top 10 from previous run

feat_with_aff = [('Previously_Insured', 'Vehicle_Damage'), ('Driving_License', 'Previously_Insured'), ('Driving_License', 'Policy_Sales_Channel'), ('Previously_Insured', 'Region_Code'), ('Previously_Insured', 'Vintage'), ('Driving_License', 'Vintage'), ('Vehicle_Damage', 'Region_Code'), ('Driving_License', 'Region_Code'), ('Vehicle_Damage', 'Policy_Sales_Channel'), ('Gender', 'Region_Code'), ('Gender', 'Policy_Sales_Channel'), ('Policy_Sales_Channel', 'Region_Code'), ('Previously_Insured', 'Policy_Sales_Channel'), ('Vintage', 'Region_Code'), ('Gender', 'Vintage'), ('Vintage', 'Policy_Sales_Channel'), ('Vehicle_Damage', 'Vintage')]
top_10_feat = feat_with_aff[:10]

cat_comb_list = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender',
                 'Vintage','Policy_Sales_Channel','Region_Code']


def top_cross(df, features):
    cross_name_list = []
    col_dict = {}
    for name1, name2 in features:
        col_dict[f'{name1}_{name2}'] = pd.factorize((df[name1].astype(str) + df[name2].astype(str)).to_numpy())[0]
        cross_name_list.append(f'{name1}_{name2}')

    df = df.assign(**col_dict)    
    # df_new = pd.DataFrame(col_dict)
    # df = pd.concat([df, df_new], axis=1).reset_index(drop=True)
    # del df_new
    return df, cross_name_list

df, cross_name_list = top_cross(df_pred_org, top_10_feat)
df[list_category+cross_name_list] = df[list_category+cross_name_list].astype(np.int8)
print(df.info())

#%% Add top10 to test

df = pd.read_parquet('data/df_test.parquet')
print(df.info())

feat_with_aff = [('Previously_Insured', 'Vehicle_Damage'), ('Driving_License', 'Previously_Insured'), ('Driving_License', 'Policy_Sales_Channel'), ('Previously_Insured', 'Region_Code'), ('Previously_Insured', 'Vintage'), ('Driving_License', 'Vintage'), ('Vehicle_Damage', 'Region_Code'), ('Driving_License', 'Region_Code'), ('Vehicle_Damage', 'Policy_Sales_Channel'), ('Gender', 'Region_Code'), ('Gender', 'Policy_Sales_Channel'), ('Policy_Sales_Channel', 'Region_Code'), ('Previously_Insured', 'Policy_Sales_Channel'), ('Vintage', 'Region_Code'), ('Gender', 'Vintage'), ('Vintage', 'Policy_Sales_Channel'), ('Vehicle_Damage', 'Vintage')]
top_10_feat = feat_with_aff[:10]

list_category = ['Vehicle_Age', 'Vintage','Policy_Sales_Channel','Region_Code']
cross_name_list = ['Previously_Insured_Vehicle_Damage', 'Driving_License_Previously_Insured', 
                   'Driving_License_Policy_Sales_Channel', 'Previously_Insured_Region_Code', 
                   'Previously_Insured_Vintage', 'Driving_License_Vintage', 
                   'Vehicle_Damage_Region_Code', 'Driving_License_Region_Code', 
                   'Vehicle_Damage_Policy_Sales_Channel', 'Gender_Region_Code']
cat_comb_list = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender',
                 'Vintage','Policy_Sales_Channel','Region_Code']


def top_cross(df, features):
    cross_name_list = []
    col_dict = {}
    for name1, name2 in features:
        col_dict[f'{name1}_{name2}'] = pd.factorize((df[name1].astype(str) + df[name2].astype(str)).to_numpy())[0]
        cross_name_list.append(f'{name1}_{name2}')

    df = df.assign(**col_dict)    

    return df, cross_name_list


df, _ = top_cross(df, top_10_feat)
df[list_category+cross_name_list] = df[list_category+cross_name_list].astype(np.int8) #df[list_category+cross_name_list].astype('category')
print(df.info())

df.to_parquet('data/test_extdata.parquet')

#%% Make small train/test set

from sklearn.model_selection import train_test_split

n = 10000
df_in = pd.read_parquet('data/train_proc.parquet')
print('shuffle')
#X = X.sample(frac=1)
df_in_y = df_in.pop('Response')

X, _,y,_ = train_test_split(df_in, df_in_y, train_size= n, 
                            random_state=27, stratify = df_in_y)

del df_in, df_in_y
y = y.to_frame()

X.to_parquet('data/train_proc_small.parquet')
y.to_parquet('data/ytrain_proc_small.parquet')



# %% Make Extended dataset

# Load data
df_train = pd.read_parquet('data/train_proc.parquet')
df_test = pd.read_parquet('data/df_test.parquet')

def add_feat(df):
    df['PI_AP'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Annual_Premium'].astype(str)).to_numpy())[0]
    df['PI_VA'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vehicle_Age'].astype(str)).to_numpy())[0]
    df['PI_VD'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vehicle_Damage'].astype(str)).to_numpy())[0]
    df['PI_V'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vintage'].astype(str)).to_numpy())[0]
    return df

df_train = add_feat(df_train)
df_test = add_feat(df_test)

list_additional = ['PI_AP', 'PI_VA', 'PI_VD', 'PI_V']
list_int32 = ['Vehicle_Age','Vintage','Policy_Sales_Channel','Region_Code']
list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
list_float32 = ['Annual_Premium','Age']
list_category = ['Vehicle_Age', 'Vintage','Policy_Sales_Channel','Region_Code']
#list_target = ['Response']

df_train[list_additional] = df_train[list_additional].astype(np.int32)
df_test[list_additional] = df_test[list_additional].astype(np.int32)

scaler = StandardScaler()
scaler.fit(df_train[list_float32])
df_test[list_float32] = scaler.transform(df_test[list_float32])

df_train.to_parquet('data/df_proc_ext.parquet')
df_test.to_parquet('data/df_test_ext_norm.parquet')

del df_train, df_test

# %%

df = pd.read_parquet('data/artifacts/train.parquet')
df = df.iloc[:100,:]
#print(df.head())
print(df.info())

for n,c in df.items():
    print('n',n)
    print('c',type(c))
    
# %%
