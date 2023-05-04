#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1672]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import missingno as msno
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats import kurtosis, skew, normaltest
import scipy
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis

from sklearn.model_selection import cross_val_score

from scipy.stats import probplot
import matplotlib.pyplot as plt


# # DATA LOADING

# In[1673]:


os.getcwd()
path="C:/Users/Bbarb/OneDrive/Desktop/..ML"
os.chdir(path)


# In[1674]:


train = pd.read_csv('illinois_basing_train_04112023.csv')
test = pd.read_csv('illinois_basing_test_04112023.csv')


# ## Investigating data

# In[1675]:


train.columns


# In[1676]:


train.shape


# In[1677]:


test.shape


# In[1678]:


train.info()


# In[1679]:


test.info()


# In[1680]:


train.describe().transpose()


# In[1681]:


test.describe().transpose()


# In[1682]:


train.std();


# In[1683]:


test.std();


# #### 'inj_diff' variable is renamed due to extra spacing in column name.

# In[1684]:


train.columns = [*train.columns[:-1], 'inj_diff']


# # DATA EXPLORATION

# #### Handling Date object: 'SampleTimeUTC' is changed from object to datetime data type. It is then indexed in both the train and test set.
# 

# In[1685]:


train['SampleTimeUTC'] = pd.to_datetime(train['SampleTimeUTC'])


# In[1686]:


test['SampleTimeUTC'] = pd.to_datetime(test['SampleTimeUTC'])


# In[1687]:


train = train.set_index('SampleTimeUTC')


# In[1688]:


test = test.set_index('SampleTimeUTC')


# ### Raw Data Time Series Plots

# ### Monitoring at the IBDP started in 2009 which is identified as the pre-injection phase and represented as the blue data in the plots. Injection at IBDP started on November 17, 2011 and is presented as orange data in the plots. The train set holds data from pre-injection and injection phases of the project. The train set holds data from the injection phase which is a continuation of the data collected 

# In[1689]:


split_date = '2011-11-17'


# In[1690]:


#Blue: Pre Injection   Orange: Injection. It can be seen by the varaiability in pressure in Zone 8 that injection has begun.

fig, axs = plt.subplots(nrows=len(train.columns), figsize=(10, 40))

for i, col in enumerate(train.columns):
    axs[i].plot(train.loc[train.index < split_date].index, train.loc[train.index < split_date, col])
    axs[i].plot(train.loc[train.index >= split_date].index, train.loc[train.index >= split_date, col])
    axs[i].set_xlabel('SampleTimeUTC')
    axs[i].set_ylabel(col, rotation=0, ha='right')

plt.tight_layout()
plt.show();


# In[1691]:


fig, axs = plt.subplots(nrows=len(test.columns), figsize=(13, 50))

for i, col in enumerate(test.columns):
    axs[i].plot(test.index, test[col], color='darkorange')
    axs[i].set_xlabel('SampleTimeUTC')
    axs[i].set_ylabel(col, rotation=0, ha='right')
    
plt.tight_layout()
plt.show();


# In[1692]:


train.hist(bins=25, figsize=(20, 20))
plt.show();


# In[1693]:


test.hist(bins=25, figsize=(20, 20))
plt.show();


# In[1694]:


fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))
axs = axs.flatten()

for i, col in enumerate(train.columns):
    sns.boxplot(x=train[col], ax=axs[i])
    axs[i].set_title(col)

for j in range(i+1, len(axs)):
    axs[j].remove()

plt.tight_layout()
plt.show();


# In[1695]:


fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))
axs = axs.flatten()

for i, col in enumerate(test.columns):
    sns.boxplot(x=test[col], ax=axs[i])
    axs[i].set_title(col)

for j in range(i+1, len(axs)):
    axs[j].remove()
    
plt.tight_layout()
plt.show();


# ### Investigating Missing Data using Missingness Matrix

# In[1696]:


msno.bar(train, color="navy", sort="ascending", figsize=(10,5), fontsize=9);


# In[1697]:


msno.bar(test, color="navy", sort="ascending", figsize=(10,5), fontsize=9);


# ##### Avg_VW1_ANPs_psi, Avg_VW1_Z03D6945Ps_psi, Avg_VW1_Z05D6720Ps_psi and Avg_VW1_Z05D6720Tp_F are missing the most values.  
# 

# In[1698]:


miss_val = train.isnull().sum()  
miss_val_pct = round(100 * train.isnull().sum() / len(train),2)
miss_val_table = pd.concat([miss_val, miss_val_pct], axis=1,  sort=True)
miss_val_table = miss_val_table[miss_val_table[0] > 0]
miss_val_table.rename(columns = {0 : 'Missing Values', 1 : '%'});


# #### In the test set, Avg_VW1_Z03D6945Ps_psi and Avg_VW1_Z03D6945Tp_F are significantlly missing in appearance.
# 

# In[1699]:


miss_val = test.isnull().sum()  
miss_val_pct = round(100 * test.isnull().sum() / len(test),2)
miss_val_table = pd.concat([miss_val, miss_val_pct], axis=1,  sort=True)
miss_val_table = miss_val_table[miss_val_table[0] > 0] 
miss_val_table.rename(columns = {0 : 'Missing Values', 1 : '%'});


# ### Counting 0 appearances

# In[1700]:


(train == 0).sum()


# In[1701]:


(test == 0).sum()


# In[1702]:


train.describe().transpose()


# In[1703]:


zone_colors = {'Avg_VW1_Z11D4917Ps_psi': 'blue',
               'Avg_VW1_Z10D5001Ps_psi': 'orange',
               'Avg_VW1_Z09D5653Ps_psi': 'green',
               'Avg_VW1_Z08D5840Ps_psi': 'red',
               'Avg_VW1_Z07D6416Ps_psi': 'purple',
               'Avg_VW1_Z06D6632Ps_psi': 'brown',
               'Avg_VW1_Z05D6720Ps_psi': 'pink',
               'Avg_VW1_Z04D6837Ps_psi': 'gray',
               'Avg_VW1_Z02D6982Ps_psi': 'olive',
               'Avg_VW1_Z01D7061Ps_psi': 'cyan',
               'Avg_VW1_Z0910D5482Ps_psi': 'magenta',
               'Avg_VW1_Z03D6945Ps_psi': 'black'}

for zone, color in zone_colors.items():
    train[zone].plot(color=color)
    
plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1704]:


zone_colors = {'Avg_VW1_Z11D4917Tp_F': 'blue',
               'Avg_VW1_Z10D5001Tp_F': 'orange',
               'Avg_VW1_Z09D5653Tp_F': 'green',
               'Avg_VW1_Z08D5840Tp_F': 'red',
               'Avg_VW1_Z07D6416Tp_F': 'purple',
               'Avg_VW1_Z06D6632Tp_F': 'brown',
               'Avg_VW1_Z05D6720Tp_F': 'pink',
               'Avg_VW1_Z04D6837Tp_F': 'gray',
               'Avg_VW1_Z02D6982Tp_F': 'olive',
               'Avg_VW1_Z01D7061Tp_F': 'cyan',
               'Avg_VW1_Z0910D5482Tp_F': 'magenta',
               'Avg_VW1_Z03D6945Tp_F': 'black'}

for zone, color in zone_colors.items():
    train[zone].plot(color=color)

plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# # DATA PREPROCESSING

# ### Predicting injection rate delta using only injection data; pre injection data is filtered out. 

# In[1705]:


train = train[(train.index > '2012-02-26')]
train = train[(train.index < '2012-03-29')]


# ### After experimentation, a filter on the train set to include observations between ‘2012-02-26’ and ‘2012-03-29’ best resembled the distribution and trends portrayed in the test set. It is important to maintain the integrity of the train set throughout the preprocessing techniques.

# In[1706]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,8))

for i, col in enumerate(train[['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F', 'Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F', 'Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi']].columns):
    ax = axes.flatten()[i]
    train[col].plot(ax=ax)
    ax.set_title(col)
    
plt.tight_layout()
plt.show();


# In[1707]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,8))

for i, col in enumerate(test[['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F', 'Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F', 'Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi']].columns):
    ax = axes.flatten()[i]
    test[col].plot(ax=ax)
    ax.set_title(col)
    
plt.tight_layout()
plt.show();


# In[1708]:


corr_train = train[train.columns[1:]].corr()['inj_diff'][:].sort_values(ascending=True).to_frame()


# In[1709]:


plt.figure(figsize = (9,9))
ax = sns.heatmap(corr_train, annot=True,cmap="YlGnBu") #seaborn heatmap
plt.title('Train Set Correlation Matrix', fontsize = 10, color = 'red');


# In[1710]:


zone_colors = {'Avg_VW1_Z11D4917Ps_psi': 'blue',
               'Avg_VW1_Z10D5001Ps_psi': 'orange',
               'Avg_VW1_Z09D5653Ps_psi': 'green',
               'Avg_VW1_Z08D5840Ps_psi': 'red',
               'Avg_VW1_Z07D6416Ps_psi': 'purple',
               'Avg_VW1_Z06D6632Ps_psi': 'brown',
               'Avg_VW1_Z05D6720Ps_psi': 'pink',
               'Avg_VW1_Z04D6837Ps_psi': 'gray',
               'Avg_VW1_Z02D6982Ps_psi': 'olive',
               'Avg_VW1_Z01D7061Ps_psi': 'cyan',
               'Avg_VW1_Z0910D5482Ps_psi': 'magenta',
               'Avg_VW1_Z03D6945Ps_psi': 'black'}

for zone, color in zone_colors.items():
    train[zone].plot(color=color)
    ax.set_xlabel('SampleTimeUTC')
    
plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1711]:


zone_colors = {'Avg_VW1_Z11D4917Ps_psi': 'blue',
               'Avg_VW1_Z10D5001Ps_psi': 'orange',
               'Avg_VW1_Z09D5653Ps_psi': 'green',
               'Avg_VW1_Z08D5840Ps_psi': 'red',
               'Avg_VW1_Z07D6416Ps_psi': 'purple',
               'Avg_VW1_Z06D6632Ps_psi': 'brown',
               'Avg_VW1_Z05D6720Ps_psi': 'pink',
               'Avg_VW1_Z04D6837Ps_psi': 'gray',
               'Avg_VW1_Z02D6982Ps_psi': 'olive',
               'Avg_VW1_Z01D7061Ps_psi': 'cyan',
               'Avg_VW1_Z0910D5482Ps_psi': 'magenta',
               'Avg_VW1_Z03D6945Ps_psi': 'black'}

for zone, color in zone_colors.items():
    test[zone].plot(color=color)

plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1712]:


zone_colors = {'Avg_VW1_Z11D4917Tp_F': 'blue',
               'Avg_VW1_Z10D5001Tp_F': 'orange',
               'Avg_VW1_Z09D5653Tp_F': 'green',
               'Avg_VW1_Z08D5840Tp_F': 'red',
               'Avg_VW1_Z07D6416Tp_F': 'purple',
               'Avg_VW1_Z06D6632Tp_F': 'brown',
               'Avg_VW1_Z05D6720Tp_F': 'pink',
               'Avg_VW1_Z04D6837Tp_F': 'gray',
               'Avg_VW1_Z02D6982Tp_F': 'olive',
               'Avg_VW1_Z01D7061Tp_F': 'cyan',
               'Avg_VW1_Z0910D5482Tp_F': 'magenta',
               'Avg_VW1_Z03D6945Tp_F': 'black'}

for zone, color in zone_colors.items():
    train[zone].plot(color=color)

plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1713]:


zone_colors = {'Avg_VW1_Z11D4917Tp_F': 'blue',
               'Avg_VW1_Z10D5001Tp_F': 'orange',
               'Avg_VW1_Z09D5653Tp_F': 'green',
               'Avg_VW1_Z08D5840Tp_F': 'red',
               'Avg_VW1_Z07D6416Tp_F': 'purple',
               'Avg_VW1_Z06D6632Tp_F': 'brown',
               'Avg_VW1_Z05D6720Tp_F': 'pink',
               'Avg_VW1_Z04D6837Tp_F': 'gray',
               'Avg_VW1_Z02D6982Tp_F': 'olive',
               'Avg_VW1_Z01D7061Tp_F': 'cyan',
               'Avg_VW1_Z0910D5482Tp_F': 'magenta',
               'Avg_VW1_Z03D6945Tp_F': 'black'}

for zone, color in zone_colors.items():
    test[zone].plot(color=color)

plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1714]:


train.isna().sum()


# In[1715]:


train.describe().transpose()


# ### Outliers

# In[1716]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,8))

for i, col in enumerate(train[['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F', 'Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F', 'Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi']].columns):
    ax = axes.flatten()[i]
    train[col].plot(ax=ax)
    ax.set_title(col)
    
plt.tight_layout()
plt.show();


# #### Replacing 0 pressure and temprature observations in all zones in train dataset and zone 3 in the test dataset.

# In[1717]:


train['Avg_VW1_Z11D4917Ps_psi'] = train['Avg_VW1_Z11D4917Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z11D4917Tp_F'] = train['Avg_VW1_Z11D4917Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z10D5001Ps_psi'] = train['Avg_VW1_Z10D5001Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z10D5001Tp_F'] = train['Avg_VW1_Z10D5001Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z09D5653Ps_psi'] = train['Avg_VW1_Z09D5653Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z09D5653Tp_F'] = train['Avg_VW1_Z09D5653Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z08D5840Ps_psi'] = train['Avg_VW1_Z08D5840Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z08D5840Tp_F'] = train['Avg_VW1_Z08D5840Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z07D6416Ps_psi'] = train['Avg_VW1_Z07D6416Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z07D6416Tp_F'] = train['Avg_VW1_Z07D6416Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z06D6632Ps_psi'] = train['Avg_VW1_Z06D6632Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z06D6632Tp_F'] = train['Avg_VW1_Z06D6632Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z05D6720Ps_psi'] = train['Avg_VW1_Z05D6720Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z05D6720Tp_F'] = train['Avg_VW1_Z05D6720Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z04D6837Ps_psi'] = train['Avg_VW1_Z04D6837Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z04D6837Tp_F'] = train['Avg_VW1_Z04D6837Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z02D6982Ps_psi'] = train['Avg_VW1_Z02D6982Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z02D6982Tp_F'] = train['Avg_VW1_Z02D6982Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z01D7061Ps_psi'] = train['Avg_VW1_Z01D7061Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z01D7061Tp_F'] = train['Avg_VW1_Z01D7061Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z0910D5482Ps_psi'] = train['Avg_VW1_Z0910D5482Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z0910D5482Tp_F'] = train['Avg_VW1_Z0910D5482Tp_F'].replace(0, np.nan)
train['Avg_VW1_Z03D6945Ps_psi'] = train['Avg_VW1_Z03D6945Ps_psi'].replace(0, np.nan)
train['Avg_VW1_Z03D6945Tp_F'] = train['Avg_VW1_Z03D6945Tp_F'].replace(0, np.nan)


# In[1718]:


test['Avg_VW1_Z03D6945Ps_psi'] = test['Avg_VW1_Z03D6945Ps_psi'].replace(0, np.nan)
test['Avg_VW1_Z03D6945Tp_F'] = test['Avg_VW1_Z03D6945Tp_F'].replace(0, np.nan)


# ## KNN Imputer

# In[1719]:


test['inj_diff'] = np.nan


# In[1720]:


# Fit KNN imputer on training set
imputer = KNNImputer(n_neighbors=5)
imputer.fit(train);


# In[1721]:


train_imputed = imputer.transform(train)
train = pd.DataFrame(train_imputed, columns=train.columns)


# In[1722]:


test_imputed = imputer.transform(test)
test = pd.DataFrame(test_imputed, columns=test.columns)
test.drop(columns=['inj_diff'], inplace=True)


# In[1723]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,8))

for i, col in enumerate(train[['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F', 'Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F', 'Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi']].columns):
    ax = axes.flatten()[i]
    train[col].plot(ax=ax)
    ax.set_title(col)
    plt.xlabel('SampleTimeUTC')

plt.tight_layout()
plt.show();


# In[1724]:


fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))
axs = axs.flatten()

for i, col in enumerate(train.columns):
    sns.boxplot(x=train[col], ax=axs[i])
    axs[i].set_title(col)

for j in range(i+1, len(axs)):
    axs[j].remove()
    
plt.tight_layout()
plt.show();


# In[1725]:


fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))
axs = axs.flatten()

for i, col in enumerate(test.columns):
    sns.boxplot(x=test[col], ax=axs[i])
    axs[i].set_title(col)

for j in range(i+1, len(axs)):
    axs[j].remove()
    
plt.tight_layout()
plt.show();


# ## IQR 

# In[1726]:


Q1 = train.quantile(0.35)
Q3 = train.quantile(0.65)
IQR = Q3 -Q1

upper_bounds = Q3 + 1.5 * IQR
train = train[(train <= upper_bounds)]
lower_bounds = Q1 - 1.5 * IQR
train = train[(train >= lower_bounds)]


# ## KNN Imputer

# In[1727]:


# Fit KNN imputer on training set
imputer = KNNImputer(n_neighbors=5)
imputer.fit(train);


# In[1728]:


train_imputed = imputer.transform(train)
train = pd.DataFrame(train_imputed, columns=train.columns)


# # DATA VISUALIZATION

# In[1729]:


fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(20, 20))
axs = axs.flatten()

for i, col in enumerate(train.columns):
    sns.boxplot(x=train[col], ax=axs[i])
    axs[i].set_title(col)
    
for j in range(i+1, len(axs)):
    axs[j].remove()
    
plt.tight_layout()
plt.show();


# In[1730]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=len(train.columns), figsize=(10, 40))

for i, col in enumerate(train.columns):
    axs[i].plot(train.index, train[col], color='darkorange')
    axs[i].set_ylabel(col, rotation=0, ha='right')
    axs[i].set_xlabel('SampleTimeUTC (hours)')
    
plt.tight_layout()
plt.show();


# In[1731]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=len(test.columns), figsize=(13, 45))

for i, col in enumerate(test.columns):
    axs[i].plot(test.index, test[col], color='darkorange')
    axs[i].set_ylabel(col, rotation=0, ha='right')
    axs[i].set_xlabel('SampleTimeUTC (hours)')
    
plt.tight_layout()
plt.show();


# In[1732]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,8))

for i, col in enumerate(train[['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F', 'Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F', 'Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi']].columns):
    ax = axes.flatten()[i]
    train[col].plot(ax=ax)
    ax.set_title(col)
    ax.set_xlabel('SampleTimeUTC (hours)')
    
plt.tight_layout()
plt.show();


# In[1733]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,8))

for i, col in enumerate(test[['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F', 'Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F', 'Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi']].columns):
    ax = axes.flatten()[i]
    test[col].plot(ax=ax)
    ax.set_title(col)
    ax.set_xlabel('SampleTimeUTC (hours)')
    
plt.tight_layout()
plt.show();


# In[1734]:


train.describe().transpose()


# In[1735]:


test.describe().transpose()


# In[1736]:


import matplotlib.pyplot as plt
import pandas as pd

cols_to_plot = ['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F','Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F','Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi','Avg_VW1_Z11D4917Ps_psi', 'Avg_VW1_Z11D4917Tp_F', 'Avg_VW1_Z10D5001Ps_psi','Avg_VW1_Z10D5001Tp_F', 'Avg_VW1_Z09D5653Ps_psi', 'Avg_VW1_Z09D5653Tp_F','Avg_VW1_Z08D5840Ps_psi', 'Avg_VW1_Z08D5840Tp_F', 'Avg_VW1_Z07D6416Ps_psi','Avg_VW1_Z07D6416Tp_F', 'Avg_VW1_Z06D6632Ps_psi', 'Avg_VW1_Z06D6632Tp_F','Avg_VW1_Z05D6720Ps_psi', 'Avg_VW1_Z05D6720Tp_F', 'Avg_VW1_Z04D6837Ps_psi','Avg_VW1_Z04D6837Tp_F', 'Avg_VW1_Z02D6982Ps_psi', 'Avg_VW1_Z02D6982Tp_F','Avg_VW1_Z01D7061Ps_psi', 'Avg_VW1_Z01D7061Tp_F', 'Avg_VW1_Z03D6945Tp_F','Avg_VW1_Z03D6945Ps_psi', 'Avg_VW1_Z0910D5482Ps_psi', 'Avg_VW1_Z0910D5482Tp_F']
fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(20,18))

for i, col in enumerate(cols_to_plot):
    row_idx = i // 5
    col_idx = i % 5
    ax = axes[row_idx, col_idx]
    ax.scatter( train['inj_diff'],train[col])
    ax.set_xlabel('inj_diff')
    ax.set_ylabel(col)
    
fig.delaxes(axes[6,3])
fig.delaxes(axes[6,4])
fig.delaxes(axes[5,3])
fig.delaxes(axes[5,4])

plt.tight_layout()
plt.show();


# In[1737]:


colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'black']

def plot_scatter(df, cols, x_col, y_label, colors):
    fig, ax = plt.subplots(figsize=(12,10))
    for i, col in enumerate(cols):
        ax.scatter(df[x_col], df[col], label=col, color=colors[i])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Zone pressures
zone_psi_cols = ['Avg_VW1_Z11D4917Ps_psi', 'Avg_VW1_Z10D5001Ps_psi', 'Avg_VW1_Z09D5653Ps_psi', 'Avg_VW1_Z08D5840Ps_psi', 'Avg_VW1_Z07D6416Ps_psi', 'Avg_VW1_Z06D6632Ps_psi', 'Avg_VW1_Z05D6720Ps_psi', 'Avg_VW1_Z04D6837Ps_psi', 'Avg_VW1_Z02D6982Ps_psi', 'Avg_VW1_Z01D7061Ps_psi', 'Avg_VW1_Z03D6945Ps_psi', 'Avg_VW1_Z0910D5482Ps_psi']
zone_psi_df = train[zone_psi_cols]
zone_psi_df['inj_diff'] = train['inj_diff']
plot_scatter(zone_psi_df, zone_psi_cols, 'inj_diff', 'Zone Pressures (psi)', colors)

# Zone temperatures
zone_temp_cols = ['Avg_VW1_Z11D4917Tp_F', 'Avg_VW1_Z10D5001Tp_F', 'Avg_VW1_Z09D5653Tp_F', 'Avg_VW1_Z08D5840Tp_F', 'Avg_VW1_Z07D6416Tp_F', 'Avg_VW1_Z06D6632Tp_F', 'Avg_VW1_Z05D6720Tp_F', 'Avg_VW1_Z04D6837Tp_F', 'Avg_VW1_Z02D6982Tp_F', 'Avg_VW1_Z01D7061Tp_F', 'Avg_VW1_Z03D6945Tp_F', 'Avg_VW1_Z0910D5482Tp_F']
zone_temp_df = train[zone_temp_cols]
zone_temp_df['inj_diff'] = train['inj_diff']
plot_scatter(zone_temp_df, zone_temp_cols,'inj_diff', 'Zone Temperatures (F)', colors)


# In[1738]:


pairs = [('Avg_VW1_Z11D4917Ps_psi', 'Avg_VW1_Z11D4917Tp_F'),
         ('Avg_VW1_Z10D5001Ps_psi', 'Avg_VW1_Z10D5001Tp_F'),
         ('Avg_VW1_Z0910D5482Ps_psi', 'Avg_VW1_Z0910D5482Tp_F'),
         ('Avg_VW1_Z09D5653Ps_psi', 'Avg_VW1_Z09D5653Tp_F'),
         ('Avg_VW1_Z08D5840Ps_psi', 'Avg_VW1_Z08D5840Tp_F'),
         ('Avg_VW1_Z07D6416Ps_psi', 'Avg_VW1_Z07D6416Tp_F'),
         ('Avg_VW1_Z06D6632Ps_psi', 'Avg_VW1_Z06D6632Tp_F'),
         ('Avg_VW1_Z05D6720Ps_psi', 'Avg_VW1_Z05D6720Tp_F'),
         ('Avg_VW1_Z04D6837Ps_psi', 'Avg_VW1_Z04D6837Tp_F'),
         ('Avg_VW1_Z03D6945Ps_psi', 'Avg_VW1_Z03D6945Tp_F'),
         ('Avg_VW1_Z02D6982Ps_psi', 'Avg_VW1_Z02D6982Tp_F'),
         ('Avg_VW1_Z01D7061Ps_psi', 'Avg_VW1_Z01D7061Tp_F')]

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(16, 10))
axs = axs.flatten()

for i, (x, y) in enumerate(pairs):
    axs[i].scatter(train[y], train[x], cmap='viridis')
    axs[i].set_xlabel(y)
    axs[i].set_ylabel(x)

fig.tight_layout()
plt.show()


# In[1739]:


pairs = [('Avg_VW1_Z11D4917Ps_psi', 'Avg_VW1_Z11D4917Tp_F'),
         ('Avg_VW1_Z10D5001Ps_psi', 'Avg_VW1_Z10D5001Tp_F'),
         ('Avg_VW1_Z0910D5482Ps_psi', 'Avg_VW1_Z0910D5482Tp_F'),
         ('Avg_VW1_Z09D5653Ps_psi', 'Avg_VW1_Z09D5653Tp_F'),
         ('Avg_VW1_Z08D5840Ps_psi', 'Avg_VW1_Z08D5840Tp_F'),
         ('Avg_VW1_Z07D6416Ps_psi', 'Avg_VW1_Z07D6416Tp_F'),
         ('Avg_VW1_Z06D6632Ps_psi', 'Avg_VW1_Z06D6632Tp_F'),
         ('Avg_VW1_Z05D6720Ps_psi', 'Avg_VW1_Z05D6720Tp_F'),
         ('Avg_VW1_Z04D6837Ps_psi', 'Avg_VW1_Z04D6837Tp_F'),
         ('Avg_VW1_Z03D6945Ps_psi', 'Avg_VW1_Z03D6945Tp_F'),
         ('Avg_VW1_Z02D6982Ps_psi', 'Avg_VW1_Z02D6982Tp_F'),
         ('Avg_VW1_Z01D7061Ps_psi', 'Avg_VW1_Z01D7061Tp_F')]

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
axs = axs.flatten()

for i, (x, y) in enumerate(pairs):
    axs[i].scatter(test[y], test[x], cmap='viridis')
    axs[i].set_xlabel(y)
    axs[i].set_ylabel(x)

fig.tight_layout()
plt.show()


# ### Pressures and Temperatures relationship of each zone, inj_diff

# In[1740]:


pairs = [('Avg_VW1_Z11D4917Ps_psi', 'Avg_VW1_Z11D4917Tp_F', 'inj_diff'),
         ('Avg_VW1_Z10D5001Ps_psi', 'Avg_VW1_Z10D5001Tp_F', 'inj_diff'),
         ('Avg_VW1_Z0910D5482Ps_psi', 'Avg_VW1_Z0910D5482Tp_F', 'inj_diff'),
         ('Avg_VW1_Z09D5653Ps_psi', 'Avg_VW1_Z09D5653Tp_F', 'inj_diff'),
         ('Avg_VW1_Z08D5840Ps_psi', 'Avg_VW1_Z08D5840Tp_F', 'inj_diff'),
         ('Avg_VW1_Z07D6416Ps_psi', 'Avg_VW1_Z07D6416Tp_F', 'inj_diff'),
         ('Avg_VW1_Z06D6632Ps_psi', 'Avg_VW1_Z06D6632Tp_F', 'inj_diff'),
         ('Avg_VW1_Z05D6720Ps_psi', 'Avg_VW1_Z05D6720Tp_F', 'inj_diff'),
         ('Avg_VW1_Z04D6837Ps_psi', 'Avg_VW1_Z04D6837Tp_F', 'inj_diff'),
         ('Avg_VW1_Z03D6945Ps_psi', 'Avg_VW1_Z03D6945Tp_F', 'inj_diff'),
         ('Avg_VW1_Z02D6982Ps_psi', 'Avg_VW1_Z02D6982Tp_F', 'inj_diff'),
         ('Avg_VW1_Z01D7061Ps_psi', 'Avg_VW1_Z01D7061Tp_F', 'inj_diff')]

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(23, 23))
axs = axs.flatten()

for i, (x, y, z) in enumerate(pairs):
    axs[i].scatter(train[y], train[x], c=train[z], cmap='viridis')
    axs[i].set_xlabel(y)
    axs[i].set_ylabel(x)
    axs[i].set_title(f'{x} vs. Temp., inj_diff')

fig.colorbar(axs[0].collections[0], shrink=0.6,location ='right', ax=axs)
plt.show()


# In[1741]:


cols_to_plotX = ['Avg_VW1_Z11D4917Tp_F','Avg_VW1_Z10D5001Tp_F',  'Avg_VW1_Z09D5653Tp_F', 'Avg_VW1_Z08D5840Tp_F', 'Avg_VW1_Z07D6416Tp_F','Avg_VW1_Z06D6632Tp_F', 'Avg_VW1_Z05D6720Tp_F', 'Avg_VW1_Z04D6837Tp_F',  'Avg_VW1_Z02D6982Tp_F', 'Avg_VW1_Z01D7061Tp_F', 'Avg_VW1_Z03D6945Tp_F', 'Avg_VW1_Z0910D5482Tp_F']
cols_to_ploty = ['Avg_VW1_Z11D4917Ps_psi', 'Avg_VW1_Z10D5001Ps_psi','Avg_VW1_Z09D5653Ps_psi','Avg_VW1_Z08D5840Ps_psi','Avg_VW1_Z07D6416Ps_psi', 'Avg_VW1_Z06D6632Ps_psi', 'Avg_VW1_Z05D6720Ps_psi','Avg_VW1_Z04D6837Ps_psi','Avg_VW1_Z02D6982Ps_psi','Avg_VW1_Z01D7061Ps_psi','Avg_VW1_Z03D6945Ps_psi', 'Avg_VW1_Z0910D5482Ps_psi']

gridsize = 30
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15,15))

for i, ax in enumerate(axs.flatten()):
    x = train['inj_diff']
    y = train[cols_to_ploty[i]]
    z = train[cols_to_plotX[i]]
    
    ax.hexbin(
        x, y, C=z, gridsize=gridsize, cmap='viridis')

    ax.set_xlabel('inj_diff')
    ax.set_ylabel(cols_to_ploty[i])
    ax.set_title('Injection Difference in ' + cols_to_plotX[i][8:-7])
    ax.invert_yaxis()

cbar_ax = fig.add_axes([1, 0.15, 0.03, 0.5])
fig.colorbar(axs[-1,-1].collections[0], cax=cbar_ax)

plt.tight_layout()
plt.show();


# In[1742]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,8))

for i, col in enumerate(train[['Avg_PLT_CO2VentRate_TPH', 'Avg_CCS1_WHCO2InjPs_psi', 'Avg_CCS1_WHCO2InjTp_F', 'Avg_CCS1_ANPs_psi', 'Avg_CCS1_DH6325Ps_psi', 'Avg_CCS1_DH6325Tp_F', 'Avg_VW1_WBTbgPs_psi', 'Avg_VW1_WBTbgTp_F', 'Avg_VW1_ANPs_psi']].columns):
    ax = axes.flatten()[i]
    train[col].plot(ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show();


# In[1743]:


zone_colors = {'Avg_VW1_Z11D4917Ps_psi': 'blue',
               'Avg_VW1_Z10D5001Ps_psi': 'orange',
               'Avg_VW1_Z09D5653Ps_psi': 'green',
               'Avg_VW1_Z08D5840Ps_psi': 'red',
               'Avg_VW1_Z07D6416Ps_psi': 'purple',
               'Avg_VW1_Z06D6632Ps_psi': 'brown',
               'Avg_VW1_Z05D6720Ps_psi': 'pink',
               'Avg_VW1_Z04D6837Ps_psi': 'gray',
               'Avg_VW1_Z02D6982Ps_psi': 'olive',
               'Avg_VW1_Z01D7061Ps_psi': 'cyan',
               'Avg_VW1_Z0910D5482Ps_psi': 'magenta',
               'Avg_VW1_Z03D6945Ps_psi': 'black'}

for zone, color in zone_colors.items():
    train[zone].plot(color=color)
    ax.set_xlabel('SampleTimeUTC')
    
plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1744]:


zone_colors = {'Avg_VW1_Z11D4917Ps_psi': 'blue',
               'Avg_VW1_Z10D5001Ps_psi': 'orange',
               'Avg_VW1_Z09D5653Ps_psi': 'green',
               'Avg_VW1_Z08D5840Ps_psi': 'red',
               'Avg_VW1_Z07D6416Ps_psi': 'purple',
               'Avg_VW1_Z06D6632Ps_psi': 'brown',
               'Avg_VW1_Z05D6720Ps_psi': 'pink',
               'Avg_VW1_Z04D6837Ps_psi': 'gray',
               'Avg_VW1_Z02D6982Ps_psi': 'olive',
               'Avg_VW1_Z01D7061Ps_psi': 'cyan',
               'Avg_VW1_Z0910D5482Ps_psi': 'magenta',
               'Avg_VW1_Z03D6945Ps_psi': 'black'}

for zone, color in zone_colors.items():
    test[zone].plot(color=color)

plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1745]:


zone_colors = {'Avg_VW1_Z11D4917Tp_F': 'blue',
               'Avg_VW1_Z10D5001Tp_F': 'orange',
               'Avg_VW1_Z09D5653Tp_F': 'green',
               'Avg_VW1_Z08D5840Tp_F': 'red',
               'Avg_VW1_Z07D6416Tp_F': 'purple',
               'Avg_VW1_Z06D6632Tp_F': 'brown',
               'Avg_VW1_Z05D6720Tp_F': 'pink',
               'Avg_VW1_Z04D6837Tp_F': 'gray',
               'Avg_VW1_Z02D6982Tp_F': 'olive',
               'Avg_VW1_Z01D7061Tp_F': 'cyan',
               'Avg_VW1_Z0910D5482Tp_F': 'magenta',
               'Avg_VW1_Z03D6945Tp_F': 'black'}

for zone, color in zone_colors.items():
    train[zone].plot(color=color)

plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1746]:


zone_colors = {'Avg_VW1_Z11D4917Tp_F': 'blue',
               'Avg_VW1_Z10D5001Tp_F': 'orange',
               'Avg_VW1_Z09D5653Tp_F': 'green',
               'Avg_VW1_Z08D5840Tp_F': 'red',
               'Avg_VW1_Z07D6416Tp_F': 'purple',
               'Avg_VW1_Z06D6632Tp_F': 'brown',
               'Avg_VW1_Z05D6720Tp_F': 'pink',
               'Avg_VW1_Z04D6837Tp_F': 'gray',
               'Avg_VW1_Z02D6982Tp_F': 'olive',
               'Avg_VW1_Z01D7061Tp_F': 'cyan',
               'Avg_VW1_Z0910D5482Tp_F': 'magenta',
               'Avg_VW1_Z03D6945Tp_F': 'black'}

for zone, color in zone_colors.items():
    test[zone].plot(color=color)

plt.legend(zone_colors.keys(), bbox_to_anchor=(1.0, 1.0));


# In[1747]:


kurtosis = train.kurtosis()
print(kurtosis)


# In[1748]:


train.hist(bins=15, figsize=(20, 20))
plt.show();


# In[1749]:


test.hist(bins=25, figsize=(20, 20))
plt.show();


# In[1750]:


train.skew()


# ### From obderving residuals, the zone pressures and tempratures show bimodial distribution as two zones of data are present.

# In[1751]:


for col in train.columns:
    fig, ax = plt.subplots()
    probplot(train[col], dist='norm', plot=ax)
    ax.set_title(f'Q-Q plot for {col}')
plt.show()
plt.tight_layout() ;


# In[1752]:


for col in test.columns:
    fig, ax = plt.subplots()
    probplot(test[col], dist='norm', plot=ax)
    ax.set_title(f'Q-Q plot for {col}')
plt.show()
plt.tight_layout();


# ### Cleaned Train Set Correlation Plot

# In[1753]:


corr_train = train[train.columns[1:]].corr()['inj_diff'][:].sort_values(ascending=True).to_frame()


# In[1754]:


corr_train


# In[1755]:


plt.figure(figsize = (9,9))

ax = sns.heatmap(corr_train, annot=True,cmap="YlGnBu")
plt.title('Train Set Correlation Matrix', fontsize = 10, color = 'red')


# ### Train- Test Split Creating Validation Set

# In[1756]:


X_train = train.drop(columns=['inj_diff'], axis =1)


# In[1757]:


y_train = train['inj_diff']


# In[1758]:


X_test = test


# In[1759]:


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)


# In[1760]:


scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[1761]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[1762]:


plt.boxplot(train['inj_diff']);


# In[1763]:


train['inj_diff'].describe()


# In[1764]:


plt.scatter(train.index, train['inj_diff'])
plt.xlabel('SampleTimeUTC')
plt.ylabel('inj_diff')
plt.xticks(rotation=90, fontsize =8)


# # MODEL: Neural Network

# In[1765]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(33,)),
    tf.keras.layers.Dense(34, activation='relu'),
    tf.keras.layers.Dense(1) ])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=0.001))
history = model.fit(X_train, y_train, epochs=100, batch_size=34, validation_split=0.2)


# In[1766]:


train_loss = model.evaluate(X_train, y_train)
val_loss = model.evaluate(X_val, y_val)
print('Training loss:', train_loss)
print('Validation loss:', val_loss)

new_data_predictions = model.predict(X_test)


# ### Training and Validation Loss Plot

# In[1767]:


train_loss = history.history['loss']
val_loss = history.history['val_loss']


plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show();


# In[1768]:


y_pred = pd.DataFrame(new_data_predictions, columns=['inj_diff'])


# In[1769]:


y_pred


# # SAVING RESULT

# In[1770]:


R = pd.DataFrame()
R['inj_diff'] = y_pred


# In[1771]:


R.to_csv('predicted_data.csv', index=False)
R
R.describe()


# In[ ]:





# In[ ]:




