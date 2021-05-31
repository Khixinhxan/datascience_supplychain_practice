#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:59:40 2020

@author: haythamomar
"""


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


time_series= pd.read_csv('timeseries.csv',parse_dates=True)
time_series.info()
time_series['date']= pd.to_datetime(time_series['date'])
time_series['date']

time_series= time_series.set_index('date')
monthly_series= time_series.total_revenue.resample('M').sum()

monthly_series.plot()


from pylab import rcParams
rcParams['figure.figsize']=16,8


components= sm.tsa.seasonal_decompose(monthly_series)
components.plot()
trend= components.trend
trend

seasonality= components.seasonal
seasonality

remainder= components.resid
remainder


monthly_series.plot()
monthly_series.rolling(window=12).mean().plot()
monthly_series.rolling(window=12).std().plot()



ad_fuller_test= sm.tsa.stattools.adfuller(monthly_series,autolag='AIC')
ad_fuller_test



from statsmodels.graphics.tsaplots import plot_acf ,plot_pacf

plot_acf(monthly_series)

plot_pacf(monthly_series)

model_ma= sm.tsa.statespace.SARIMAX(monthly_series,order= (0,0,1))
results_ma= model_ma.fit()

results.aic


model_AR= sm.tsa.statespace.SARIMAX(monthly_series,order= (1,0,0))
results_AR= model_AR.fit()

results_AR.aic


model_ARma= sm.tsa.statespace.SARIMAX(monthly_series,order= (1,0,1))
results_ARma= model_ARma.fit()

results_ARma.aic


model_ARima= sm.tsa.statespace.SARIMAX(monthly_series,order= (1,1,1))
results_ARima= model_ARima.fit()

results_ARima.aic

results_ARima.plot_diagnostics(figsize=(15, 12))


import itertools       
           
P=D=Q=p=d=q= range(0,3)
S= 12
combinations= list(itertools.product(p,d,q,P,D,Q))
len(combinations)
arima_orders=[(x[0],x[1],x[2]) for x in combinations]
arima_orders[0][0]
seasonal_orders=[(x[3],x[4],x[5],S) for x in combinations]

results_data= pd.DataFrame(columns=['p','d','q','P','D','Q','AIC'])

### length of combinatioons

len(combinations) 

for i in range(len(combinations)):
     try:
      
          model = sm.tsa.statespace.SARIMAX(monthly_series,order=arima_orders[i],
                                        seasonal_order= seasonal_orders[i]
                                       )
          result= model.fit()
          results_data.loc[i,'p']= arima_orders[i][0]
          results_data.loc[i,'d']= arima_orders[i][1]
          results_data.loc[i,'q']= arima_orders[i][2]
          results_data.loc[i,'P']= seasonal_orders[i][0]
          results_data.loc[i,'D']= seasonal_orders[i][1]
          results_data.loc[i,'Q']= seasonal_orders[i][2]
          results_data.loc[i,'AIC']= result.aic
     except:
          continue
      
                                   
     
results_data[results_data.AIC == min(results_data.AIC)]     
     
best_model= sm.tsa.statespace.SARIMAX(monthly_series,order=(0,1,0),
                                      seasonal_order= (0,2,1,12))

results=  best_model.fit()

monthly_series
fitting= results.get_prediction(start= '2009-12-31')
fitting_mean= fitting.predicted_mean
forecast= results.get_forecast(steps=12)
forcast_mean= forecast.predicted_mean

fitting_mean.plot(label='Fitting')
monthly_series.plot(label='Actual')
forecast_mean.plot(label='Forecast')
plt.legend(loc='upperleft')

mean_absolute_error= abs(monthly_series-fitting_mean).mean()
rmse_best_model= np.sqrt((monthly_series-fitting_mean)**2).mean()
#### arima

model_ARima= sm.tsa.statespace.SARIMAX(monthly_series,order= (1,1,1))
results_ARima= model_ARima.fit()

results_ARima.aic

fitted_arima= results_ARima.get_prediction(start='2009-12-31')
fitting_arima= fitting.predicted_mean

mae_arima= abs(monthly_series- fitting_arima).mean()
rmse_arima= np.sqrt((monthly_series-fitting_arima)**2).mean()

import statsmodels as sm
sm.tsa.holtwinters.ExponentialSmoothing

model_expo1= sm.tsa.holtwinters.ExponentialSmoothing(monthly_series,trend='add',
                                                     seasonal='add',seasonal_periods=12)
model_expo2= sm.tsa.holtwinters.ExponentialSmoothing(monthly_series,trend='mul',
                                                     seasonal='add',seasonal_periods=12)
model_expo3= sm.tsa.holtwinters.ExponentialSmoothing(monthly_series,trend='add',
                                                     seasonal='mul',seasonal_periods=12)

model_expo4= sm.tsa.holtwinters.ExponentialSmoothing(monthly_series,trend='mul',
                                                     seasonal='mul',seasonal_periods=12)

results_1= model_expo1.fit()
results_2= model_expo2.fit()
results_3= model_expo3.fit()
results_4= model_expo4.fit()

results_1.summary()
results_2.summary()
results_3.summary()
results_4.summary()

fit1= model_expo1.fit().predict(0,len(monthly_series))
fit2= model_expo2.fit().predict(0,len(monthly_series))
fit3= model_expo3.fit().predict(0,len(monthly_series))
fit4= model_expo4.fit().predict(0,len(monthly_series))

mae1= abs(monthly_series- fit1).mean()
mae2= abs(monthly_series- fit2).mean()
mae3= abs(monthly_series- fit3).mean()
mae4= abs(monthly_series- fit4).mean()

forecast=model_expo1.fit().predict(0,len(monthly_series)+12)

monthly_series.plot(label='actual')
forecast.plot(label='forecast')
plt.legend(loc='upperleft')




























