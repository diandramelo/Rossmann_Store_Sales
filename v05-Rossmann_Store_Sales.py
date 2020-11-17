#!/usr/bin/env python
# coding: utf-8

# # 0. IMPORTS

# ## 0.1. Libraries

# In[81]:


import pandas as pd
import inflection
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display       import Image
from matplotlib            import gridspec
from tabulate              import tabulate
from scipy.stats           import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
from boruta                import BorutaPy
from sklearn.ensemble      import RandomForestRegressor

pd.options.mode.chained_assignment = None  # default='warn'


# ## 0.2. Helper Functions

# In[2]:


def cramer_v(x,y):
    cm = pd.crosstab(x, y).values
    n = cm.sum()
    r, k = cm.shape
    chi2 = chi2_contingency (cm )[0]
    
    chi2corr = max(0, chi2 - (k-1)*(r-1)/(n-1))
    
    kcorr = k - (k-1)**2/(n-1)
    
    rcorr = r - (r-1)**2/(n-1)
    
    return np.sqrt( (chi2corr/n) / (min(kcorr-1, rcorr-1 )))


# ## 0.3. Loading Data

# In[3]:


df_sales_raw = pd.read_csv('data/train.csv', low_memory=False)
df_store_raw = pd.read_csv('data/store.csv', low_memory=False)

# merge
df_raw = pd.merge(df_sales_raw, df_store_raw, how='left', on = 'Store')


# ## 0.4. Graph Visualization

# In[4]:


from IPython.core.display import HTML

def jupyter_settings():
   get_ipython().run_line_magic('matplotlib', 'inline')
   get_ipython().run_line_magic('pylab', 'inline')
   plt.style.use( 'bmh' )
   plt.rcParams['figure.figsize'] = [18, 8]
   plt.rcParams['font.size'] = 20
   display( HTML( '<style>.container { width:100% !important; }</style>') )
   pd.options.display.max_columns = None
   pd.options.display.max_rows = None
   pd.set_option( 'display.expand_frame_repr', False )
   sns.set()
    
jupyter_settings()


# # 1. DESCRIBE DATA

# In[5]:


df1 = df_raw.copy()


# ## 1.1. Rename Columns

# In[6]:


# list(df1.columns)

cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday',
            'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

# CamelCase to SnakeCase
snakecase = lambda x: inflection.underscore(x)

cols_new = list( map( snakecase, cols_old ) )

# rename
df1.columns = cols_new


# ## 1.2. Data Dimensions

# In[7]:


print('Number of Rows: {}'.format(df1.shape[0]))
print('Number of Columns: {}'.format(df1.shape[1]))


# ## 1.3. Data Types

# In[8]:


df1.sample()


# In[9]:


df1['date'] = pd.to_datetime(df1['date'])
df1.dtypes


# ## 1.4. Check NA

# In[10]:


df1.isna().sum()


# ## 1.5. Fillout NA

# In[11]:


# competition_distance (2642 NAs)

# df1['competition_distance'].max() = 75860.0

df1['competition_distance'] = df1['competition_distance'].fillna(200000)


# competition_open_since_month (323348 NAs)
df1['competition_open_since_month'] = df1['competition_open_since_month'].fillna(df1['date'].dt.month)
# similarly: df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )


# competition_open_since_year (323348 NAs)
df1['competition_open_since_year'] = df1['competition_open_since_year'].fillna(df1['date'].dt.year)


# promo2_since_week (508031 NAs)
df1['promo2_since_week'] = df1['promo2_since_week'].fillna(df1['date'].dt.week)


# promo2_since_year (508031 NAs)
df1['promo2_since_year'] = df1['promo2_since_year'].fillna(df1['date'].dt.year)


# promo_interval (508031 NAs)
month_map = {1: 'Jan',
             2: 'Feb',
             3: 'Mar',
             4: 'Apr',
             5: 'May',
             6: 'Jun',
             7: 'Jul',
             8: 'Aug',
             9: 'Sep',
             10: 'Oct',
             11: 'Nov',
             12: 'Dec'}

df1['promo_interval'].fillna(0, inplace=True )

df1['month_map'] = df1['date'].dt.month.map( month_map )

# New variable: check if sales date was in the period of promotion
df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 )


# ## 1.6. Change Data Type

# In[12]:


# competition
df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
    
# promo2
df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

df1.dtypes


# ## 1.7. Descriptive Statistics

# In[13]:


num_attributes = df1.select_dtypes(include = ['int64', 'float64'])
cat_attributes = df1.select_dtypes(exclude = ['int64', 'float64', 'datetime64[ns]'])


# ### 1.7.1. Numerical Variables

# In[14]:


# Central Tendency - Mean, Median
ct1 = pd.DataFrame(num_attributes.mean())
ct2 = pd.DataFrame(num_attributes.median())

# Dispersion - Std, Min, Max, Range, Skew, Kurtosis
d1 = pd.DataFrame(num_attributes.std())
d2 = pd.DataFrame(num_attributes.min())
d3 = pd.DataFrame(num_attributes.max())
d4 = pd.DataFrame(num_attributes.max() - num_attributes.min())
d5 = pd.DataFrame(num_attributes.skew())
d6 = pd.DataFrame(num_attributes.kurtosis())

# concat
df_num = pd.concat([ct1, ct2, d1, d2, d3, d4, d5, d6], axis = 1)
df_num.columns = ['mean', 'median', 'std', 'min', 'max', 'range', 'skew', 'kurtosis']
df_num


# In[15]:


sns.displot(df1['competition_distance'], kde = False);


# ### 1.7.2. Categorical Variables

# In[16]:


cat_attributes.apply( lambda x: x.unique().shape[0])


# In[17]:


aux1 = df1[(df1['state_holiday'] != '0') & (df1['sales'] > 0)] # filtering data for better visualization

plt.subplot(1,3,1)
sns.boxplot(x = 'state_holiday', y = 'sales', data = aux1);

plt.subplot(1,3,2)
sns.boxplot(x = 'store_type', y = 'sales', data = aux1);

plt.subplot(1,3,3)
sns.boxplot(x = 'assortment', y = 'sales', data = aux1);


# # 2. FEATURE ENGINEERING

# In[18]:


df2 = df1.copy()


# ## 2.1. Hypothesis Mind Map

# In[19]:


Image(filename='Daily_Store_Sales.png')


# ## 2.2. Hypothesis Development

# Creation of hypothesis for each category in the Mind Map, while related to the response variable << **sales** >> .
# 
# PS.: Clients & Location: none information on dataset

# ### 2.2.1. Store Hypothesis

# **1.** Stores with **more diversified products** should sell **more**
# 
# **2.** Stores with **closer competitors** should sell **less**

# ### 2.2.2. Product Hypothesis

# **1.** Stores with **longer active promotions** should sell **more**
# 
# **2.** Stores with **more consecutive promotions** should sell **more**

# ### 2.2.3. Time Series Hypothesis

# **1.** Stores, during **school holidays**, should sell **more**
# 
# **2.** Stores, in **Christmas period**, should sell **more** than during other holidays
# 
# **3.** Stores, during **weekends**, should sell **less**
# 
# **4.** At the **first half of the month**, stores should sell **more**
# 
# **5** In the **first semester of the year**, stores should sell **less**
# 
# **6.** **Throughout the years**, stores should sell **more**.

# ## 2.3. Feature Engineering

# ### 2.3.1. Created Columns

# In[20]:


# year
df2['year'] = df2['date'].dt.year

# month
df2['month'] = df2['date'].dt.month

# day
df2['day'] = df2['date'].dt.day

# week_of_year
df2['week_of_year'] = df2['date'].dt.week

# year_week
df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

###

# competition_since
df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year = x['competition_open_since_year'], month = x['competition_open_since_month'], day = 1), axis = 1)
df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)

# promo_since
df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days = 7))
df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)


# ### 2.3.2. Modified Columns

# In[21]:


# state_holiday
df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

# assortment
df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')


# # 3. FILTERING VARIABLES

# In[22]:


df3 = df2.copy()


# ## 3.1. Filtering Rows

# In[23]:


df3 = df3[(df3['open'] != 0) & (df3['sales'] > 0)]


# ## 3.2. Columns Selection

# In[24]:


cols_drop = ['customers', 'open', 'competition_open_since_month' , 'competition_open_since_year', 'promo2_since_week',
             'promo2_since_year', 'promo_interval', 'month_map']
df3 = df3.drop(cols_drop, axis=1)


# # 4. EXPLORATORY DATA ANALYSIS (EDA)

# In[25]:


df4 = df3.copy()


# ## 4.1. Univariate Analysis

# ### 4.1.1. Response Variable

# In[26]:


sns.distplot(df4['sales']);


# ### 4.1.2. Numerical Variables

# In[27]:


num_attributes.hist(bins = 25);


# ### 4.1.3. Categorical Variables

# In[28]:


# state_holiday
aux1 = df4[df4['state_holiday'] != 'regular_day']

plt.subplot(3, 2, 1)
sns.countplot(aux1['state_holiday']);

plt.subplot(3, 2, 2)
sns.kdeplot(df4[df4['state_holiday'] == 'public_holiday']['sales'], shade = True, label = 'public_holiday');
sns.kdeplot(df4[df4['state_holiday'] == 'easter_holiday']['sales'], shade = True, label = 'easter_holiday');
sns.kdeplot(df4[df4['state_holiday'] == 'christmas']['sales'], shade = True, label = 'christmas');
plt.legend();


# store_type
plt.subplot(3, 2, 3)
sns.countplot(df4['store_type']);

plt.subplot(3, 2, 4)
sns.kdeplot(df4[df4['store_type'] == 'a']['sales'], shade = True, label = 'a');
sns.kdeplot(df4[df4['store_type'] == 'b']['sales'], shade = True, label = 'b');
sns.kdeplot(df4[df4['store_type'] == 'c']['sales'], shade = True, label = 'c');
sns.kdeplot(df4[df4['store_type'] == 'd']['sales'], shade = True, label = 'd');
plt.legend();


# assortment
plt.subplot(3, 2, 5)
sns.countplot(df4['assortment']);

plt.subplot(3, 2, 6)
sns.kdeplot(df4[df4['assortment'] == 'basic']['sales'], shade = True, label = 'basic');
sns.kdeplot(df4[df4['assortment'] == 'extra']['sales'], shade = True, label = 'extra');
sns.kdeplot(df4[df4['assortment'] == 'extended']['sales'], shade = True, label = 'extended');
plt.legend();


# ## 4.2. Bivariate Analysis - Hypothesis Validation

# ### 4.2.1. Individual Analysis of the Hypothesis

# **H1.** Stores with **more diversified products** should sell **more**
# 
# **TRUE**, when compared to basic assortment, although extended assortment does not sell more than extra

# In[29]:


aux1 = df4[['assortment', 'sales']].groupby('assortment').mean().reset_index()

plt.subplot(1, 2, 1)
sns.barplot(x = 'assortment', y = 'sales', data = aux1);


# **H2.** Stores with **closer competitors** should sell **less**
# 
# **FALSE** - sales do not vary much

# In[30]:


aux1 = df4[['competition_distance', 'sales']].groupby('competition_distance').mean().reset_index()

plt.subplot(1, 3, 1)
bins = list(np.arange(0,20000,1000))
aux1['competition_distance_binned'] = pd.cut(aux1['competition_distance'], bins = bins)
aux2 = aux1[['competition_distance_binned', 'sales']].groupby('competition_distance_binned').mean().reset_index()
sns.barplot(x = 'competition_distance_binned', y = 'sales', data = aux2);
plt.xticks(rotation = 90);

plt.subplot(1, 3, 2)
sns.scatterplot(x = 'competition_distance', y = 'sales', data = aux1);

plt.subplot(1, 3, 3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True);


# **H3.** Stores with **longer active promotions** should sell **more**
# 
# **TRUE** - There is a tendency of more average sales for stores with longer periods of promotion

# In[31]:


aux1 = df4[['promo_time_week', 'sales']].groupby('promo_time_week').mean().reset_index()

grid = gridspec.GridSpec(2,3)


# extended promotion
aux2 = aux1.loc[(aux1['promo_time_week'] >= 0), :]

plt.subplot(grid[0,0])
bins = list(np.arange(0,313,20))
aux2['promo_time_week_binned'] = pd.cut(aux2['promo_time_week'], bins = bins)
aux3 = aux2[['promo_time_week_binned', 'sales']].groupby('promo_time_week_binned').mean().reset_index()
sns.barplot(x = 'promo_time_week_binned', y = 'sales', data = aux3);
plt.xticks(rotation = 90);

plt.subplot(grid[0,1])
sns.regplot(x = 'promo_time_week', y = 'sales', data = aux2);


# regular promotion
aux4 = aux1[aux1['promo_time_week'] < 0]

plt.subplot(grid[1,0])
bins = list(np.arange(-126,0,5))
aux4['promo_time_week_binned'] = pd.cut(aux4['promo_time_week'], bins = bins)
aux5 = aux4[['promo_time_week_binned', 'sales']].groupby('promo_time_week_binned').mean().reset_index()
sns.barplot(x = 'promo_time_week_binned', y = 'sales', data = aux5);
plt.xticks(rotation = 90);

plt.subplot(grid[1,1])
sns.regplot(x = 'promo_time_week', y = 'sales', data = aux4);


plt.subplot(grid[:,2])
sns.heatmap(aux1.corr(method = 'pearson'), annot = True);


# **H4.** Stores with **more consecutive promotions** should sell **more**
# 
# **FALSE** - Stores with regular promotions only sell more

# In[32]:


df4[['promo', 'promo2', 'sales']].groupby(['promo', 'promo2']).mean().reset_index()


# In[33]:


aux1 = df4[(df4['promo'] == 1) & (df4['promo2'] == 1)][['year_week', 'sales']].groupby('year_week').mean().reset_index()
ax = aux1.plot();

aux2 = df4[(df4['promo'] == 1) & (df4['promo2'] == 0)][['year_week', 'sales']].groupby('year_week').mean().reset_index()
aux2.plot(ax=ax);

ax.legend(labels=['Regular + Extended', 'Regular']);


# **H5.** Stores, during **school holidays**, should sell **more**
# 
# **TRUE**, except for the months of January, April, November and December

# In[34]:


aux1 = df4[['school_holiday', 'sales']].groupby('school_holiday').mean().reset_index()

plt.subplot(2, 1, 1)
sns.barplot(x = 'school_holiday', y = 'sales', data = aux1);

# analysis per month
plt.subplot(2, 1, 2)
aux2 = df4[['school_holiday', 'month', 'sales']].groupby(['school_holiday', 'month']).mean().reset_index()
sns.barplot(x = 'month', y = 'sales', hue = 'school_holiday', data = aux2);


# **H6.** Stores, in **Christmas period**, should sell **more** than during other holidays
# 
# **FALSE** - Stores sell less in Christmas than in public holiday or easter

# In[35]:


# 2015 removed - data is only until 31/07/2015

aux1 = df4[(df4['state_holiday'] != 'regular_day') & (df4['year'] < 2015)]

aux2 = aux1[['state_holiday', 'year', 'sales']].groupby(['year', 'state_holiday']).mean().reset_index()

sns.barplot(x = 'state_holiday', y = 'sales', hue = 'year', data = aux2);


# **H7.** Stores, during **weekends**, should sell **less**
# 
# **TRUE**

# In[36]:


aux1 = df4[['day_of_week', 'sales']].groupby('day_of_week').sum().reset_index()

plt.subplot(1, 2, 1)
sns.barplot(x = 'day_of_week', y = 'sales', data = aux1);

plt.subplot(1, 2, 2)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True);


# **H8.** At the **first half of the month**, stores should sell **more**
# 
# **TRUE**

# In[37]:


aux1 = df4[['day', 'sales']].groupby('day').sum().reset_index()

plt.subplot(1, 3, 1)
sns.barplot(x = 'day', y = 'sales', data = aux1);
plt.xticks(rotation = 90);

plt.subplot(1, 3, 2)
sns.regplot(x = 'day', y = 'sales', data = aux1);

plt.subplot(1, 3, 3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True);


# **H9.** In the **first semester of the year**, stores should sell **less**
# 
# **TRUE** - Analysis for the years of 2013 and 2014 (the data from 2015 stops in july)

# In[38]:


# 2015 removed - data only until 31/07/2015

aux = df4[df4['year'] < 2015]

aux1 = aux[['month', 'sales']].groupby('month').sum().reset_index()

plt.subplot(1, 3, 1)
sns.barplot(x = 'month', y = 'sales', data = aux1);

plt.subplot(1, 3, 2)
sns.regplot(x = 'month', y = 'sales', data = aux1);

plt.subplot(1, 3, 3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True);


# **H10.** **Throughout the years**, stores should sell **more**
# 
# **FALSE** - Even though data of 2015 is incomplete, it was considered to observe the tendency of sales reduction throughout years

# In[39]:


aux1 = df4[['year', 'sales']].groupby('year').sum().reset_index()

plt.subplot(1, 3, 1)
sns.barplot(x = 'year', y = 'sales', data = aux1);

plt.subplot(1, 3, 2)
sns.regplot(x = 'year', y = 'sales', data = aux1);

plt.subplot(1, 3, 3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True);


# ### 4.2.2. General Hypothesis Review

# In[40]:



tab = [['Hypothesis', 'Conclusion', 'Relevance'],
       ['H1 - Stores with more diversified products should sell more', 'True', 'Medium'],
       ['H2 - Stores with closer competitors should sell less', 'False', 'Low'],
       ['H3 - Stores with longer active promotions should sell more', 'True', 'Medium'],
       ['H4 - Stores with more consecutive promotions should sell more', 'False', 'Low'],
       ['H5 - Stores, during school holidays, should sell more', 'True', 'Low'],
       ['H6 - Stores, in Christmas period, should sell more than during other holidays', 'False', 'Medium'],
       ['H7 - Stores, during weekends, should sell less', 'True', 'High'],
       ['H8 - At the first half of the month, stores should sell more', 'True', 'High'],
       ['H9 - In the first semester of the year, stores should sell less', 'True', 'High'],
       ['H10 - Throughout the years, stores should sell more', 'False', 'High']
      ]
print(tabulate(tab, headers='firstrow'))


# ## 4.3. Multivariate Analysis

# ### 4.3.1. Numerical Attributes

# In[41]:


sns.heatmap(num_attributes.corr(method = 'pearson'), annot = True);


# ### 4.3.2. Categorical Attributes

# In[42]:


a = df4.select_dtypes(include='object')

# calculating cramer v
a1 = cramer_v(a['state_holiday'], a['state_holiday'])
a2 = cramer_v(a['state_holiday'], a['store_type'])
a3 = cramer_v(a['state_holiday'], a['assortment'])

a4 = cramer_v(a['store_type'], a['state_holiday'])
a5 = cramer_v(a['store_type'], a['store_type'])
a6 = cramer_v(a['store_type'], a['assortment'])

a7 = cramer_v(a['assortment'], a['state_holiday'])
a8 = cramer_v(a['assortment'], a['store_type'])
a9 = cramer_v(a['assortment'], a['assortment'])

# final dataset
d = pd.DataFrame({'state_holiday':[a1,a2,a3], 
                  'store_type':[a4,a5,a6],
                  'assortment':[a7,a8,a9] })

d = d.set_index(d.columns)
sns.heatmap(d, annot=True);


# # 5. DATA PREPARATION

# In[43]:


df5 = df4.copy()


# ## 5.1. Normalization

# Technique that works well for variables that have a normal distribution. As this does not occur in the numerical variables presented, this technique **will not be applied in this case**.

# ## 5.2. Rescaling

# In[44]:


mms = MinMaxScaler()
rs = RobustScaler()

# competition_distance (Robust Scaler)
df5['competition_distance'] = rs.fit_transform(df5[['competition_distance']].values)

# competition_time_month
df5['competition_time_month'] = rs.fit_transform(df5[['competition_time_month']].values)

# promo_time_week
df5['promo_time_week'] = mms.fit_transform(df5[['promo_time_week']].values)

# year
df5['year'] = mms.fit_transform(df5[['year']].values)


# ## 5.3. Transformation

# ### 5.3.1. Response Variable

# In[45]:


df5['sales'] = np.log1p(df5['sales'])


# ### 5.3.2. Encoding

# In[46]:


#state_holiday - One Hot Encoder
df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

#store_type - Label Encoder
le = LabelEncoder()
df5['store_type'] = le.fit_transform(df5['store_type'])

# assortment - Ordinal Encoder
assort_dict = {'basic': 1,
               'extra': 2,
               'extended': 3}
df5['assortment'] = df5['assortment'].map(assort_dict)


# ### 5.3.3. Nature Transformation

# In[47]:


# day_of_week
df5['day_of_week_sin'] = np.sin(2 * np.pi * df5['day_of_week'] / 7) 
df5['day_of_week_cos'] = np.cos(2 * np.pi * df5['day_of_week'] / 7) 

# day
df5['day_sin'] = np.sin(2 * np.pi * df5['day'] / 30) 
df5['day_cos'] = np.cos(2 * np.pi * df5['day'] / 30) 

# month
df5['month_sin'] = np.sin(2 * np.pi * df5['month'] / 12) 
df5['month_cos'] = np.cos(2 * np.pi * df5['month'] / 12) 

# week_of_year
df5['week_of_year_sin'] = np.sin(2 * np.pi * df5['week_of_year'] / 52) 
df5['week_of_year_cos'] = np.cos(2 * np.pi * df5['week_of_year'] / 52) 


# # 6. FEATURE SELECTION

# In[62]:


df6 = df5.copy()


# In[63]:


# removing colinear variables

cols_drop = ['day_of_week', 'day', 'month', 'week_of_year', 'year_week', 'competition_since', 'promo_since']

df6 = df6.drop(cols_drop, axis = 1)

df6.columns


# ## 6.1. Split dataframe into training and test dataset

# In[78]:


df6['date'].max() - pd.Timedelta(value = 6, unit = 'W')


# In[79]:


# training dataset
X_train = df6[df6['date'] < '2015-06-19']
Y_train = X_train['sales']

print('Training Min Date: {}'.format(X_train['date'].min()))
print('Training Max Date: {}'.format(X_train['date'].max()))


# test dataset
X_test = df6[df6['date'] >= '2015-06-19']
Y_test = X_test['sales']

print('\nTest Min Date: {}'.format(X_test['date'].min()))
print('Test Max Date: {}'.format(X_test['date'].max()))


# ## 6.2. Boruta as Feature Selector

# In[ ]:


# conversion of df/series to numpy and remotion of variables that do not compose training ('sales' and 'date')
X_train_n = X_train.drop(['sales', 'date'], axis=1).values
Y_train_n = Y_train.values.ravel()

# define RandomForestRegressor
rf = RandomForestRegressor(n_jobs = -1)

# define Boruta
boruta = BorutaPy(rf, n_estimators = 'auto', verbose = 2, random_state = 42).fit(X_train_n, Y_train_n)


# ### 6.2.1. Best Features

# In[ ]:


cols_selected = boruta.support_.tolist()

# best features
X_train_fs = X_train.drop(['sales', 'date'], axis = 1)
cols_selected_boruta = X_train_fs.iloc[:, cols_selected].columns.tolist()

# colunas desconsideradas pelo boruta
cols_not_selected_boruta = list(np.setdiff1d(X_train_fs.columns, cols_selected_boruta))


# ## 6.3. Saving Boruta Best Features
