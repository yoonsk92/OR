import pandas as pd
import numpy as np
import os
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import seaborn
import scipy.stats as scp
import datetime as dt

os.chdir('C:/Users/yoons/Desktop/York/OMIS4000 Models and Applications of OR/Project/Daily Orders')

df = pd.read_excel('SBD.xlsx')
df = df.drop(df.index[[range(6)]])
df = df.drop(df.columns[0], axis = 1)
df = df.drop(df.columns[2:26], axis = 1)
df.columns = ['Date', 'Orders']
df = df.reset_index(drop = True)
df = df.drop(df.index[[1084,1085,1086]])
df['Date'] = pd.to_datetime(df['Date'])
df['Orders'] = pd.to_numeric(df['Orders'])
df = df.set_index('Date')
df = df.asfreq('d')
print(df.index.freq)

#time series of daily orders (Feb 03,2015 ~ Feb 01, 2018)
df.plot()

#Seasonal decomposition, monthly = There is seasonality.
df.interpolate(inplace = True)
sm.tsa.seasonal_decompose(df).plot()


#0 = Monday
wk = df.groupby([df.index.dayofweek],0).sum()
wk.index = ['Mon','Tues','Wed','Thur','Fri','Sat','Sun']
wk


fifth = (df.index >= '2015-02-03') & (df.index <= '2015-12-31')
fifth_ = df.loc[fifth]
fifth_avg = fifth_.groupby(lambda x: x.dayofweek).mean()
fifth_avg.columns = ['Orders for 2015']
fifth_avg.index = ['Mon','Tues','Wed','Thur','Fri','Sat','Sun']

sixth = (df.index >= '2016-01-02') & (df.index <= '2016-12-31')
sixth_ = df.loc[sixth]
sixth_avg = sixth_.groupby(lambda x: x.dayofweek).mean()
sixth_avg.columns = ['Orders for 2016']

seventh = (df.index >= '2017-01-01') & (df.index <= '2017-12-31')
seventh_ = df.loc[seventh]
seventh_avg = seventh_.groupby(lambda x: x.dayofweek).mean()
seventh_avg.columns = ['Orders for 2017']

eighth = (df.index >= '2018-01-02') & (df.index <= '2018-02-01')
eighth_ = df.loc[eighth]
eighth_avg = eighth_.groupby(lambda x: x.dayofweek).mean()
eighth_avg.columns = ['Orders for 2018']

#Its clear that Saturday is the busiest day of the week
ax = fifth_avg.plot()
sixth_avg.plot(ax=ax)
seventh_avg.plot(ax=ax)
eighth_avg.plot(ax=ax)

#Test for stationarity, its stationary.
def test_stationarity(timeseries):
    rolmean = pd.rolling_mean(timeseries, window = 7)
    rolstd = pd.rolling_std(timeseries, window = 7)
    
    orig = plt.plot(timeseries, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block = False)

#Shows stationary
test_stationarity(df)

#Statistical test show they are stationary
dftest = adfuller(df['Orders'], autolag = 'AIC')
dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Numberof Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s) ' % key] = value
print (dfoutput)

#Fit AR(1) == series is stionary but highly seasonal
ar = AR(df['Orders'])
res_ar = ar.fit(maxlag = 1)
res_ar.params


#Arima
#ACF
plot_acf(df['Orders'], lags = 120)
lag_acf = acf(df['Orders'], nlags = 120)
plt.plot(lag_acf)
plt.axhline(y = 0, linestyle = '--', color = 'blue')
plt.axhline(y = 1.96/np.sqrt(len(df['Orders'])), linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(df['Orders'])), linestyle = '--', color = 'gray')

plot_pacf(df['Orders'], lags = 120)
lag_pacf = pacf(df['Orders'], nlags = 120)
plt.plot(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'blue')
plt.axhline(y = 1.96/np.sqrt(len(df['Orders'])), linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(df['Orders'])), linestyle = '--', color = 'gray')

#Training (2015-02-03 to 2017-02-01)
math.floor(len(df.index) * 0.7)
train = df[df.index <= df.index[math.floor(len(df.index) * 0.7)]]
test = df[df.index > '2017-03-10']

train_test = sm.tsa.statespace.SARIMAX(train, order = (4,1,1),
                                       seasonal_order = (0,1,0,365),
                                       enforce_stationarity = False,
                                       enforce_invertibility = False)

########################################################
train_test_fit = train_test.fit()
train_test_forecast = train_test_fit.forecast(len(test))
plt.plot(test, label = 'Actual')
plt.plot(train_test_forecast, color = 'orange', label = 'Forecasts')
plt.legend()
plt.show()

forecasts = sm.tsa.statespace.SARIMAX(df, order = (4,1,1),
                                      seasonal_order = (0,1,0,365),
                                      enforce_stationarity = False,
                                      enforce_invertibility = False)

forecasts_fit = forecasts.fit()
forecasts_x = forecasts_fit.forecast(548)
plt.plot(df)
plt.plot(forecasts_x, label = 'Forecasts')

from pandas import ExcelWriter
writer = ExcelWriter('test.xlsx')
test.to_excel(writer)
writer.save()
test.to_excel

##########################################################
seaborn.boxplot(df.index.dayofweek, df.Orders)
seaborn.boxplot(df.Orders, orient = "v")

loc1, scale1 = scp.norm.fit(df.Orders)
shape2, loc2, scale2 = scp.gamma.fit(df.Orders)
shape3, loc3, scale3 = scp.lognorm.fit(df.Orders, floc=0)
loc4, scale4 = scp.expon.fit(df.Orders)
loc5, scale5 = scp.logistic.fit(df.Orders, floc=0)

import scipy
size = 150
x = scipy.arange(size)

plt.hist(df.Orders, bins=10, normed=True)
plt.plot(scp.norm.pdf(x, loc1, scale1), label="Norm")
plt.plot(scp.gamma.pdf(x, shape2, loc2, scale2), label="Gamma")
plt.plot(scp.lognorm.pdf(x, shape3, loc3, scale3), label="Lognormal")
plt.plot(scp.expon.pdf(x, loc4, scale4), label="Exponential")
plt.plot(scp.logistic.pdf(x, loc5, scale5), label="Logistic")
plt.legend(['Normal', 'Gamma', 'Lognormal', 'Exponential', 'Logistic'], loc='upper right')
plt.title("Histogram of the Orders")
plt.ylabel('Number of Orders')
plt.xlabel('Service Time Duration (Seconds)');
plt.xlim(0, 1000)
plt.ylim(0, 0.008)
plt.show()

scp.probplot(x, (loc1, scale1), dist="norm", plot=plt)
ax1 = plt.subplot(221)
scp.probplot(x, (shape2, loc2, scale2), dist="gamma", plot=plt)
ax1 = plt.subplot(222)
scp.probplot(x, (shape3, loc3, scale3), dist="lognorm", plot=plt)
ax1 = plt.subplot(223)
scp.probplot(x, (loc4, scale4), dist="expon", plot=plt)
ax1 = plt.subplot(224)
scp.probplot(x, (loc5, scale5), dist="logistic", plot=plt)
plt.show()

scp.ttest_1samp(scp.norm.pdf(x, loc1, scale1), np.average(df.Orders))
scp.ttest_1samp(scp.gamma.pdf(x, shape2, loc2, scale2), np.average(df.Orders))
scp.ttest_1samp(scp.lognorm.pdf(x, shape3, loc3, scale3), np.average(df.Orders))
scp.ttest_1samp(scp.expon.pdf(x, loc4, scale4), np.average(df.Orders))
scp.ttest_1samp(scp.logistic.pdf(x, loc5, scale5), np.average(df.Orders))

scp.kstest(scp.norm.pdf(x, loc1, scale1), 'norm')
scp.kstest(scp.gamma.pdf(x, shape2, loc2, scale2), 'gamma', (shape2,))
scp.kstest(scp.lognorm.pdf(x, shape3, loc3, scale3), 'lognorm', (shape3,))
scp.kstest(scp.expon.pdf(x, loc4, scale4), 'expon')
scp.kstest(scp.logistic.pdf(x, loc5, scale5), 'logistic')            

#REMOVE OUTLIER = BECAME NORMAL
mean = np.mean(df.Orders, axis = 0)
sd = np.std(df.Orders, axis = 0)            
final_list = [x for x in df.Orders if (x > mean - 2 * sd)]      
final_list = [x for x in final_list if (x < mean + 2 * sd)]          
print (final_list)
plt.boxplot(final_list)

loc1, scale1 = scp.norm.fit(final_list)
shape2, loc2, scale2 = scp.gamma.fit(final_list)
shape3, loc3, scale3 = scp.lognorm.fit(final_list, floc=0)
loc4, scale4 = scp.expon.fit(final_list)
loc5, scale5 = scp.logistic.fit(final_list, floc=0)

import scipy
size = 80
x = scipy.arange(size)
plt.hist(final_list, bins=30, normed=True)
plt.plot(scp.norm.pdf(x, loc1, scale1), label="Norm")
plt.plot(scp.gamma.pdf(x, shape2, loc2, scale2), label="Gamma")
plt.plot(scp.lognorm.pdf(x, shape3, loc3, scale3), label="Lognormal")
plt.plot(scp.expon.pdf(x, loc4, scale4), label="Exponential")
plt.plot(scp.logistic.pdf(x, loc5, scale5), label="Logistic")
plt.legend(['Normal', 'Gamma', 'Lognormal', 'Exponential', 'Logistic'], loc='upper right')
plt.title("Histogram of the Orders")
plt.ylabel('Number of Orders')
plt.xlabel('Service Time Duration (Seconds)');
plt.xlim(0, 1000)
plt.ylim(0, 0.008)
plt.show()

scp.probplot(x, (loc1, scale1), dist="norm", plot=plt)
ax1 = plt.subplot(221)
scp.probplot(x, (shape2, loc2, scale2), dist="gamma", plot=plt)
ax1 = plt.subplot(222)
scp.probplot(x, (shape3, loc3, scale3), dist="lognorm", plot=plt)
ax1 = plt.subplot(223)
scp.probplot(x, (loc4, scale4), dist="expon", plot=plt)
ax1 = plt.subplot(224)
scp.probplot(x, (loc5, scale5), dist="logistic", plot=plt)
plt.show()

zz = df.groupby(lambda x: x.dayofweek).mean()
zz.columns = ['Aggregate Orders 95% CI']
zz.index = ['Mon', 'Tues','Wed','Thur','Fri','Sat','Sun']
zz.plot()


#SALES BY TIME
df = pd.read_excel('SBT.xlsx')
df = df.drop(df.index[[8,9,10,11,12]], axis = 0)
df = df.drop(df.index[[0]])
df = df.drop(df.columns[0], axis = 1)
df = df.dropna(axis = 1, how = 'all')
df = df.reset_index(drop = True)
df = df.T
df.columns = ['Status Name', 'Booked', 'Cancelled', 'Checked-in', 'Confirmed', 'No Show', 'Paid/Complete']
df = df.drop(df.index[[0]])
df['Status Name'] = pd.DatetimeIndex(df['Status Name'], format = '%H:%M:%S')
df1 = df.groupby(df['Status Name'].dt.hour).sum()

_max = df1['Paid/Complete'].max()
twentyfive = df1['Paid/Complete'].max() * 0.25

high = (df1['Paid/Complete'] >= _max - twentyfive)
highdf = df1[high]

mid = (df1['Paid/Complete'] >= twentyfive) & (df1['Paid/Complete'] <= _max - twentyfive)
middf = df1[mid]

low = (df1['Paid/Complete'] <= twentyfive)
lowdf = df1[low]


#OPTIMIZE
import pulp
model = pulp.LpProblem('Employee Scheduling', pulp.LpMinimize)

I = list(range(1,15))
S = list(range(1,12))
nFT = list(range(1,8))
nPT = list(range(1,2))
FT_pay = 15
PT_pay = 15
x_FT = {}
x_PT = {}
        
for nft in nFT:
    for s in S:
        x_FT[nft,s] = pulp.LpVariable('x_FT(%s,%s)' % (nft,s), cat = 'Binary')

for npt in nPT:
    for s in S:
        x_PT[npt,s] = pulp.LpVariable('x_PT(%s,%s)' % (npt,s), cat = 'Binary')

full_time_cost = pulp.lpSum(FT_pay * x_FT[nft,s] for nft in nFT for s in S)
part_time_cost = pulp.lpSum(PT_pay * x_PT[npt,s] for npt in nPT for s in S)
model += full_time_cost + part_time_cost, 'Cost'

#Constraints for highly demanded hours
for a in range(8):
    model += pulp.lpSum(x_FT[nft,s] for nft in nFT) + pulp.lpSum(x_PT[npt,s] for npt in nPT) >= highdf.iloc[a,5]

#Constraints for mildly demanded hours
for b in range(3):
    model += pulp.lpSum(x_FT[nft,s] for nft in nFT) + pulp.lpSum(x_PT[npt,s] for npt in nPT) >= middf.iloc[b,5]

#Constraints for low demanded hours
for c in range(3):
    model += pulp.lpSum(x_FT[nft,s] for nft in nFT) + pulp.lpSum(x_PT[npt,s] for npt in nPT) >= lowdf.iloc[c,5]




























