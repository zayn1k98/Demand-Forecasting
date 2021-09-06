import pandas as pd 
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from scipy import stats 
import statsmodels.api as sm 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from matplotlib import rcParams
from pylab import rcParams
import statsmodels.api as sm
import itertools
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from tkinter import *  
from tkinter import messagebox 

df = pd.read_csv('Historical Product Demand.csv', parse_dates=['Date'])
df.head(10)
df.dtypes
print(df.isnull().any().sum(), ' / ', len(df.columns))
# Check any number of data points with NaN
print(df.isnull().any(axis=1).sum(),'/', len(df))

#Lets check which column has null values.
print (df.isna().sum())

print ('Null to Dataset Ratio for "Dates" Column '': ',df.isnull().sum()[3]/df.shape[0]*100)

df.dropna(axis=0, inplace=True) #Remove all the rows with na's
df.reset_index(drop=True)
df.sort_values('Date')[1:50]

df['Order_Demand']=df['Order_Demand'].str.replace('(',"")
df['Order_Demand']=df['Order_Demand'].str.replace(')',"")
df.head(100)
#Since the "()" has been removed , Now i Will change the data type.

df['Order_Demand'] = df['Order_Demand'].astype('int64')
df['Date'].min() , df['Date'].max()

# figure size in inches
rcParams['figure.figsize'] = 10,5

sn.distplot(df['Order_Demand'],fit=norm)

#Get the QQ-plot
fig = plt.figure()
res = stats.probplot(df['Order_Demand'], plot=plt)
plt.show()

df['Warehouse'].value_counts().sort_values(ascending=False)
df.groupby('Warehouse').sum().sort_values('Order_Demand', ascending = False)
print(len(df['Product_Category'].value_counts()))
rcParams['figure.figsize'] = 50,14
sn.countplot(df['Product_Category'].sort_values(ascending = True))


rcParams['figure.figsize']=20,5 #Figure Size in Inches for Plotting
f, axes = plt.subplots(1,2)

regDataWH=sn.boxplot(df['Warehouse'],df['Order_Demand'],ax=axes[0]) #Create a variable for Regular Data for WH and OD 

logDataWH=sn.boxplot(df['Warehouse'],np.log1p(df['Order_Demand']),ax=axes[1]) #Craete a Variable with Log Transformation

del regDataWH, logDataWH

#Step-02: Check the Order Demand Qty by Product Category (PC)
rcParams['figure.figsize']=20,5
f,axes =plt.subplots(1,2)

regDataPC=sn.boxplot(df['Product_Category'],df['Order_Demand'],ax=axes[0])
logDataPC=sn.boxplot(df['Product_Category'],df['Order_Demand'],ax=axes[1])

del regDataPC, logDataPC

df=df.groupby('Date')['Order_Demand'].sum().reset_index()
#Step-02: Indexing the Date Column as for further procssing.
df = df.set_index('Date')
df.index #Lets check the index
#Step-03:#Averages daily sales value for the month, and we are using the start of each month as the timestamp.
monthly_avg_sales = df['Order_Demand'].resample('MS').mean()
#In case there are Null values, they can be imputed using bfill.
monthly_avg_sales = monthly_avg_sales.fillna(monthly_avg_sales.bfill())
#Visualizing time series.

monthly_avg_sales.plot(figsize=(20,10))
plt.show()

rcParams['figure.figsize'] = 20, 10
decomposition = sm.tsa.seasonal_decompose(monthly_avg_sales, model='additive')
fig = decomposition.plot()
plt.show()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX1: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX2: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX3: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX4: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(monthly_avg_sales,
                                            order=param,
                                            seasonal_order=param_seasonal,enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
#ARIMAX WITH SARIMAX MODEL
mod = sm.tsa.statespace.SARIMAX(monthly_avg_sales,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(20, 10))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=False)
pred_ci = pred.conf_int()

#Plotting real and forecasted values.
ax = monthly_avg_sales['2016':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

#forecasting
y_forecasted = pred.predicted_mean
y_truth = monthly_avg_sales['2016-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE {}'.format(round(mse, 2)))
print('RMSE: {}'.format(round(np.sqrt(mse), 2)))

pred_uc = results.get_forecast(steps=75)
pred_ci = pred_uc.conf_int()
ax = monthly_avg_sales.plot(label='observed', figsize=(16, 8))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

'''XGBoost'''


#read only part of data
data=pd.read_csv('Historical Product Demand.csv',nrows=100000)
df.dropna(axis=0)
predictors=['Product_Code','Warehouse','Product_Category','Date']
X=data[predictors]
y=data.Order_Demand
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1,test_size=0.1,train_size=0.5)
one_hot_encoding_val_data=pd.get_dummies(val_X)
one_hot_encoding_train_data=pd.get_dummies(train_X)
del val_X
del train_X
final_train, final_val = one_hot_encoding_train_data.align(one_hot_encoding_val_data,join='left',axis=1)
#final val and final train are X 
del one_hot_encoding_val_data
del one_hot_encoding_train_data

my_imputer = SimpleImputer()
final_train = my_imputer.fit_transform(final_train)
final_val = my_imputer.transform(final_val)
#define model 
'''XGBoost'''
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(final_train, train_y, verbose=False)
del final_train
del train_y
# make predictions
predictions = my_model.predict(final_val)
del final_val
print("XGBoost Mean Absolute Error : " + str(mean_absolute_error(predictions, val_y)))
top=Tk()
top.geometry("10x10")
messagebox.showinfo("Business Intelligence ","profit is:70% and Loss is:30%")
top.mainloop()