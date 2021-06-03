#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:23:19 2020

@author: haythamomar
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
retail_clean= pd.read_csv('retail_clean.csv')

retail_clean.info()
retail_clean.InvoiceDate
retail_clean['InvoiceDate']= pd.to_datetime(retail_clean['InvoiceDate'])
retail_clean['date']= retail_clean['InvoiceDate'].dt.strftime("%Y-%m-%d")
retail_clean['date']=pd.to_datetime(retail_clean['date'])

retail_clean['month']= retail_clean.date.dt.month
retail_clean['year']= retail_clean.date.dt.year
retail_clean['week']= retail_clean.date.dt.week

retail_clean.columns
retail_clean.month.describe()
time_series=retail_clean.groupby(['week','year']).agg(date= ('date','first'),
        total_sales=('Quantity',np.sum)).reset_index().sort_values('date')


sns.lineplot(x='date',y='total_sales',data=time_series)




from sklearn.linear_model import LinearRegression

time_series['trend']= range(time_series.shape[0])
time_series['month']= time_series['date'].dt.month

####dropping columns

X= time_series.drop(['week','year','date','total_sales'],axis=1)

names=pd.get_dummies(X).columns
X= pd.get_dummies(X).values
y= time_series.total_sales.values


model= LinearRegression()

model.fit(X,y)

model.get_params()
model.coef_

dict1= list(zip(names,model.coef_))


prediction= model.predict(X)

time_series['prediction']= prediction
import matplotlib.pyplot as plt

plt.plot(time_series.date,time_series.total_sales,label='Actual')
plt.plot(time_series.date,time_series.prediction,label='prediction')
plt.legend(loc='upperleft')
plt.show()

#####forecasting

time_series=retail_clean.groupby(['week','month','year']).agg(date= ('date','first'),
        total_sales=('Quantity',np.sum)).reset_index().sort_values('date')


max_date= time_series.date.max()

dates= pd.DataFrame({'date':pd.date_range('2011-12-12','2012-08-5',freq='W')})

time_series= pd.concat([time_series,dates],axis=0)

time_series['trend']= range(time_series.shape[0])
time_series['month']= time_series['date'].dt.month
time_series['month']= time_series['month'].astype('category')

####dropping columns

X= time_series.drop(['week','year','date','total_sales'],axis=1)

names=pd.get_dummies(X).columns
X= pd.get_dummies(X).values
y= time_series.total_sales.values

prediction= model.predict(X)

time_series['prediction']= prediction

plt.plot(time_series.date,time_series.total_sales,label='Actual')
plt.plot(time_series.date,time_series.prediction,label='prediction')
plt.legend(loc='upperleft')
plt.show()










