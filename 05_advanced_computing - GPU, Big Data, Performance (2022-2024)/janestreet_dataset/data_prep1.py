#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gc

#%% Load one time series from all parquet files

df_0 = pd.DataFrame
for i in range(10):
    print(i)
    df = pd.read_parquet(f'data/raw/train.parquet/partition_id={i}')
    if i == 0:
        df_0 = df.loc[df.symbol_id == 0]
        del df
        gc.collect()
    if i > 0:
        df_0 = pd.concat([df_0, df.loc[df.symbol_id == 0]], axis=0)
        del df
        gc.collect()
        
print(df_0.shape)
df_0.to_parquet('data/artifacts/train_symbol_0.parquet')

# %% Load one parquet file and check data reduction


def reduce_mem_usage(df):
    #memory_usage, B->KB->MB
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)!='category':
            c_min,c_max = df[col].min(),df[col].max() 
            if str(col_type)[:3] == 'int': #int8,int16,int32, int64
                #(-128 x 127)
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                #(-32,768 x 32,767)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                #(-2,147,483,648 x2,147,483,647)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                #(-9,223,372,036,854,775,808 x 9,223,372,036,854,775,807)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #float32
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


df_1 = pd.read_parquet(f'data/raw/train.parquet/partition_id=0')
df_red = reduce_mem_usage(df_1)




# %%
df_1 = pd.read_parquet(f'data/raw/train.parquet/partition_id=0')
print(df_1.info())


# %%

print(df.info())

# %%
