#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet('data/artifacts/train.parquet')
print(df.info())

# %%

df_1 = df.loc[df.Response == 1]
df_0 = df.loc[df.Response == 0]

#%%

age_stat = pd.concat([df.Age.value_counts().sort_index(), df_1.Age.value_counts().sort_index(),
                      df_0.Age.value_counts().sort_index()], axis=1)
age_stat.columns = ['org_count', 'resp1_count', 'resp0_count']
age_stat['rel1'] = age_stat.resp1_count / age_stat.org_count
age_stat['rel0'] = age_stat.resp0_count / age_stat.org_count

#display(age_stat.head())

plt.subplot(1,2,1)
plt.bar(age_stat.index, age_stat.rel1.values)
plt.title('Repsons=1')
plt.subplot(1,2,2)
plt.bar(age_stat.index, age_stat.rel0.values)
plt.title('Response=0')
plt.show()

# %%

print(len(df.Annual_Premium.unique()))

plt.figure()
plt.boxplot(df.Annual_Premium)
plt.show()

# %%

plt.figure()
plt.hist(df.Annual_Premium, bins=1000)
plt.show()


# %%
