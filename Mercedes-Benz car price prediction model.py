#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from statistics import median
from scipy.stats import iqr

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


# ### Problem statement - Mercedes-Benz car price prediction model (best possible features for good predictive model)
#      - engine_capacity
#      - types of fuel (diesel cars are costly; )
#      - color /paint of the car
#      - no. of seaters
#      - mileage 
#      - model type
#      - safety features (low/medium/large)
#      - date of selling (seasonality)
#      - IsAutomative (yes/no)
#      - 
#      --- second hand cars
#      - distance travelled 
#      - mileage 
#      - age of the car
#      - warrenty flag
#      - insurance flag
#      - condition (poor/ok/good)

# In[3]:


df = pd.read_csv('Mercedes_Benz_pricing_challenge.csv')
print(df.shape)
# df.info()


# In[4]:


df.describe()


# In[5]:


df.head()


# ### EDA

# In[6]:


## duplicate check
df.duplicated().sum()


# In[7]:


df.duplicated(['model_key', 'mileage', 'engine_power',
       'registration_date', 'fuel', 'paint_color', 'car_type', 'feature_1',
       'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
       'feature_7', 'feature_8', 'price', 'sold_at']).sum()


# ## Encoding 
#  - OHE (usage: nominal)
#  - Label Encoding (usage: ordinal, range, intervals )   

# In[8]:


bool_f = df.select_dtypes('boolean').columns

for col in bool_f:
    print(col, df[col].unique())


# In[9]:


for col in bool_f:
    df[col] = df[col].apply(lambda x: 1 if x==True else 0)


# In[10]:


df[bool_f]


# In[11]:


categorical_f = df.select_dtypes('object').columns


# In[12]:


categorical_f = [ 'model_key', 'fuel', 'paint_color','car_type']


# In[13]:


df[categorical_f]


# In[14]:


# distinct values for each categorical features
for col in categorical_f:
   print(col, df[col].unique(), '=>', df[col].nunique())


# In[17]:


## if assigns (1,2,3,4) for encoding fuel ; which level should be assigned with what value?
# fuel ['diesel' 'petrol' 'hybrid_petrol' 'electro'] [3,2,4,1]


# #### Encoding 'fuel' types

# In[18]:


### plot frequency of each of the fuel types and five point summary on price for each fuel type
fig, ax = plt.subplots(1,2,figsize = (15,4))
sns.countplot(df.fuel, ax = ax[0])
sns.boxplot(x ='price',y = 'fuel', data = df, ax = ax[1])
plt.show()


### get plot normalised metrices on price for each fuel type
fuel_PU = []
for each in df.fuel.unique():
    x = df[df.fuel==each]
    fuel_PU.append(median(x['price'])) # getting median

fig, ax = plt.subplots(1,2,figsize = (15,4))
sns.barplot(fuel_PU, df.fuel.unique(), ax = ax[0])
ax[0].set_xlabel('Median Price')
# ax[1].set_ylabel('Fuel Type')

fuel_PU = []
for each in df.fuel.unique():
    x = df[df.fuel==each]
    fuel_PU.append(np.mean(x['price']))

sns.barplot(fuel_PU, df.fuel.unique(), ax = ax[1])
ax[1].set_xlabel('Avg. Price')
# ax[2].set_ylabel('Fuel Type')

plt.show()


# - Quick observations:
#     - Median price of hybrid-petrol cars are highest amongst all types, while petrol cars have lowest median price. If median is to be considered as deciding metric then, order of weightage for each fuel types follows:
#         - {'diesel':2,'petrol':1,'hybrid_petrol':4, 'electro':3}
#     - Average price for each fuel type suggest weightage as: 
#         - {'diesel':2,'petrol':1,'hybrid_petrol':4, 'electro':3}
#     - Negligible outliers impact on price by fuel type
#     - Hence, fuel can be Encoded with the weightage assignment as follows:
#      - {'diesel':2,'petrol':1,'hybrid_petrol':4, 'electro':3}

# In[19]:


df['fuel_E'] = df.fuel.map({'diesel':2,'petrol':1,'hybrid_petrol':4, 'electro':3})


# In[20]:


### plot frequency of each of the paint_color and five point summary on price for each paint_color  type
fig, ax = plt.subplots(1,2,figsize = (15,4))
sns.countplot(df.paint_color , ax = ax[0]).set_title('Frequency of car paint color')
sns.boxplot(x ='price',y = 'paint_color', data = df, ax = ax[1])
plt.show()


### get plot normalised metrices on price for each paint_color  type
paint_color_PU = []
for each in df.paint_color .unique():
    x = df[df.paint_color ==each]
    paint_color_PU.append(median(x['price'])) # getting median

fig, ax = plt.subplots(1,2,figsize = (15,4)) 
sns.barplot(sorted(paint_color_PU), df.paint_color.unique()[np.argsort(paint_color_PU)], ax = ax[0])
ax[0].set_xlabel('Median Price')
# ax[1].set_ylabel('paint_color  Type')

paint_color_PU = []
for each in df.paint_color .unique():
    x = df[df.paint_color ==each]
    paint_color_PU.append(np.mean(x['price']))

sns.barplot(sorted(paint_color_PU), df.paint_color.unique()[np.argsort(paint_color_PU)], ax = ax[1])
ax[1].set_xlabel('Avg. Price')
# ax[2].set_ylabel('paint_color  Type')

plt.show()


# In[21]:


paint_col_each= [np.mean(df[df.paint_color==each]['price']) for each in df.paint_color.unique()]
# dummy dataframe
x = pd.DataFrame(df.paint_color.unique(), columns=['paint_color'])
x['total_avgPrice']  = df['price'].mean()
x['avgPrice_by_color'] = paint_col_each
x['diff_avgPrice'] = (x.total_avgPrice - x.avgPrice_by_color)/x.total_avgPrice*100
x.sort_values('diff_avgPrice')


# - Quick observations:
#     - Orange cars have almost same median and average price and highest amost all.
#     - Green cars have least median and average price.
#     - Other colors of the cars median and average prices are changing in order.
#     - All the cars paint colors except orange, white, silver and green, average price is within 5% of the total price.
#  - following the observations below weightage can be used for label encoding
#         - orange -> 5
#         - white -> 4
#         - remaing -> 3
#         - silver -> 2
#         - green -> 1
# 

# In[22]:


df.model_key.nunique()


# In[23]:


df['paint_color_E'] = df.paint_color.apply(lambda x : 5 if x == 'orange' else (
4 if x == 'white' else (
2 if x == 'silver' else (
1 if x == 'green' else 3))))


# In[24]:


model_key_PU = []
for each in df.model_key.unique():
    x = df[df.model_key==each]
    model_key_PU.append(np.mean(x['price']))
    
len(model_key_PU)    
plt.figure(figsize = (15,20))
x = pd.concat([pd.DataFrame(df.model_key.unique(), columns=['Key']), pd.DataFrame(model_key_PU,columns=['value'])], axis = 1).sort_values('value')
sns.barplot(x='value',y='Key' ,data = x)
plt.xlabel('Avg. Price')
plt.ylabel('Model Keys')


# In[25]:


df['model_key_E'] = df.model_key.apply(lambda x : 6 if x == 'i8' else (
5 if x == 'M4' else (
4 if x == 'X6 M' else (
1 if x == 'X5 M50' else (
1 if x in ('735', '216', '523', '650', '123', 'Z4', '118', '116', '316',
       '630', '318', '114', '220 Active Tourer', '320', '120', '125',
       '216 Active Tourer', 'X1') else (
2 if x in ( '325', '318 Gran Turismo', '525',
       '218 Active Tourer', '520', '218 Gran Tourer', '518', '328', '330',
       '216 Gran Tourer', '218', '320 Gran Turismo', '214 Gran Tourer',
       'X3', '225', '225 Active Tourer', '635', '530', '520 Gran Turismo',
       '528', '325 Gran Turismo', '418 Gran Coupé', '530 Gran Turismo',
       'ActiveHybrid 5', 'i3', '335', '135') else (
3)))))))


# In[26]:


car_type = []
for each in df.car_type.unique():
    x = df[df.car_type==each]
    car_type.append(np.mean(x['price']))
    
x = pd.concat([pd.DataFrame(df.car_type.unique(), columns=['Key']), pd.DataFrame(car_type,columns=['value'])], axis = 1).sort_values('value', ascending= False)
plt.figure(figsize = (10,4))
sns.barplot(x='value',y='Key' ,data = x)
plt.xlabel('Price')
plt.ylabel('Car Types')
plt.show()


# In[27]:


df['car_type_E'] = df.car_type.apply(lambda x : 1 if x == 'subcompact' else (
2 if x == 'estate' else (
3 if x == 'hatchback' else (
4 if x == 'van' else (
5 if x == 'sedan' else (
6 if x == 'convertible' else (
7 if x == 'suv' else 8)))))))


# #### Capturing interraction effects between features

# In[28]:


# ### relation between model_key and mileage
mileage_N = []
for each in df.model_key.unique():
    x = df[df.model_key==each]
    mileage_N.append(sum(x['mileage'])/len(x))
    
x = pd.concat([pd.DataFrame(df.model_key.unique(), columns=['Key']), pd.DataFrame(mileage_N,columns=['value'])], axis = 1).sort_values('value', ascending= False)
plt.figure(figsize = (10,15))
sns.barplot(x='value',y='Key' ,data = x)
plt.xlabel('Mileage')
plt.ylabel('Model Key')


# In[29]:


df['model_key_mileage_rel'] = df.model_key.apply(lambda x : 5 if x == '523' else (
4 if x in ('530 Gran Turismo', '735', '635', '525', '335', '530') else (
3 if x in ('535', '325 Gran Turismo', '518', '520', '320',
       '220 Active Tourer', '730', '318', 'M5', '630', '318 Gran Turismo',
       '316', 'M550', '520 Gran Turismo', '335 Gran Turismo', '118', 'X3',
       '120', '330', '320 Gran Turismo', 'X5') else (
2 if x in ('X1', '740', '325', '328',
       'X6', '528', '640 Gran Coupé', '123', '116', '135', 'M3', 'X5 M',
       '750', '640', '418 Gran Coupé', '216 Active Tourer',
       '430 Gran Coupé', '125', 'Z4') else (
1)))))


# In[30]:


fig, ax = plt.subplots(2,2,figsize = (20,8))
sns.distplot(df.engine_power, ax = ax[0,0])
sns.distplot(df.mileage, ax = ax[0,1], color = 'green')
sns.scatterplot(df.mileage, df.price , ax = ax[1,1],color = 'green')
sns.scatterplot(df.engine_power, df.price , ax = ax[1,0])


# In[31]:


### bucketing price in range of 100
df['EP_bucket'] = df.engine_power.apply(lambda x: '[100)' if x < 100 else ('[100-200)'  if x < 200 else ('[200-300)' if x < 300 else ('[300-400)]' if x < 400 else '[400)') )))
x = df.groupby('EP_bucket')['price'].mean().reset_index()
sns.barplot(x = x.EP_bucket, y = x.price).set_ylabel('Avg. price')
plt.show()
x = df.groupby('EP_bucket')['price'].count().reset_index()
sns.barplot(x = x.EP_bucket, y = x.price).set_ylabel('Count')
plt.show()
df.groupby('EP_bucket')['price'].describe()


# - Note: engine_power is already numerical form and can be feed as a feature in raw form. 
#     - more analysis can be done to find any peculiar point from the data 

# ### target analysis

# In[32]:


fig, ax = plt.subplots(1,3,figsize = (15,4))
sns.distplot(df.price, color = 'green', ax = ax[0])
sns.distplot(np.log1p(df.price), color = 'blue', ax = ax[1])
sns.distplot(np.sqrt(df.price), color = 'c', ax = ax[2])
plt.show()


# In[33]:


## correlaation 
df[['mileage', 'engine_power','price']].corr()


# ### date related features analysis

# In[34]:


df.registration_date = pd.to_datetime(df.registration_date)
df.sold_at = pd.to_datetime(df.sold_at)
df['gap_reg_sold_year'] =  df.sold_at.dt.year - df.registration_date.dt.year 
df['reg_month'] = df.registration_date.dt.month
df['gap_reg_sold_days'] =  df.sold_at - df.registration_date 
df['gap_reg_sold_days'] = df.gap_reg_sold_days.apply(lambda x: int(str(x).split()[0]))


# In[35]:


df.head()


# In[36]:


# ### hypothesis: How does the estimated value of a car change over time?
x = df.groupby('gap_reg_sold_year')['price'].mean().reset_index()
sns.lineplot(x.gap_reg_sold_year, x.price ,color = 'green').set_xlabel('year diff bet/ registration and sold date')
plt.show()


# In[37]:


## correlaation 
df[['mileage', 'engine_power','price','gap_reg_sold_year']].corr()


# In[38]:


### drop records where mileage is negative and engine_power is 0
df.drop(df[df.mileage < 0].index, axis = 0, inplace = True)
df.drop(df[df.engine_power == 0].index, axis = 0, inplace = True)


# In[39]:


df.head()


# In[40]:


# regression problem  
#  -  linear regression
#  -  RFR
#  -  SVR


# #### regression problem  
# linear regression
#     - Adv. (simple)
#     - Disadv. 
#         - works work if features are linearly related to that of the target
#         - does not work well with  many categorical features
#         - does not wotk when Hetroscedasticity  
# RFR
#     - Adv.
#         - works well if features are non-linear
#     - Disadv. 
#         - are highly unstable 
#         - training time will higher
# SVR
#     - Adv. 
#         - works with all time data (linear, non-linear, etc) given that you choose right kernal
#     - Disadv. 
#         - highly complex model (training time take is high)
#         - choosing right kernal is must

# ## Modelling
# 
# 

# In[50]:


# import packages
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_log_error


# In[51]:


features = ['mileage', 'engine_power', 'feature_1',
       'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
       'feature_7', 'feature_8', 'fuel_E', 'paint_color_E',
       'model_key_E', 'car_type_E', 'model_key_mileage_rel','gap_reg_sold_year']

target = 'price'


# In[52]:


## splitting data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size = 0.2, random_state = 42 )


# In[53]:


# Age -> 20 - 55
# salary -> 20,000 - 30,00,000


# In[54]:


### Model with default parameters; features scaling has not happened
RFModel = RandomForestRegressor().fit(X_train,y_train) # training
## 
print('Train R^2 score {}'.format(r2_score(y_train, RFModel.predict(X_train))))
print('Test R^2 score {}'.format(r2_score(y_test, RFModel.predict(X_test))))

print('Train mean_squared_log_error {}'.format(mean_squared_log_error(y_train, RFModel.predict(X_train))))
print('Test mean_squared_log_error {}'.format(mean_squared_log_error(y_test, RFModel.predict(X_test))))


# In[55]:


## scaling features
scaler = StandardScaler().fit(X_train,y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[56]:


## Get r-suqare score with each models
for model in [LinearRegression(), Ridge(), RandomForestRegressor(),SVR()]:
    scores = cross_val_score(model, X_train, y_train, cv= 5, scoring='r2',n_jobs=-1)
    print('Model =>',str(model))
    print('Avg. R^2_score: {}, std:{}, scores:{}'.format(np.mean(scores),np.std(scores), scores))


# In[57]:


### Model with default parameters; features are scaled
RFModel = RandomForestRegressor(random_state=42).fit(X_train,y_train) # training
## 
print('Test R^2 score {}'.format(r2_score(y_test, RFModel.predict(X_test))))
print('Test mean_squared_log_error {}'.format(mean_squared_log_error(y_test, RFModel.predict(X_test))))


# In[58]:


### Model with default parameters; features are scaled
RFModel = RandomForestRegressor(oob_score=True,random_state=42).fit(X_train,y_train) # training
## 
# print('Test R^2 score {}'.format(r2_score(y_train, RFModel.predict(X_train))))
print('Test R^2 score {}'.format(r2_score(y_test, RFModel.predict(X_test))))
print('Test mean_squared_log_error {}'.format(mean_squared_log_error(y_test, RFModel.predict(X_test))))


# In[ ]:





# In[ ]:




