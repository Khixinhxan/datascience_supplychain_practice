#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:37:00 2020

@author: haythamomar
"""
import pandas as pd

retail= pd.read_csv('twentyeleven.csv')

retail.info()

retail['InvoiceDate']= pd.to_datetime(retail['InvoiceDate'])
retail['daysofweek']= retail['InvoiceDate'].dt.dayofweek

retail['daysofweek'].value_counts()
retail['date']= retail['InvoiceDate'].dt.strftime('%Y-%m-%d')
retail['date']=pd.to_datetime(retail['date'])


#### CV2 and the average demand interval

retail_grouped=retail.groupby(['Description','date']).agg(total_sales=('Quantity','sum')).reset_index()

cv_data= retail_grouped.groupby('Description').agg(average=('total_sales','mean'),
                                                   sd=('total_sales','std')).reset_index()

cv_data['cv_squared']= (cv_data['sd']/cv_data['average'])**2



#### the average demand interval per product 


product_by_date=retail.groupby(['Description','date']).agg(count=('Description','count')).reset_index()
retail[['Description','date']].drop_duplicates()


skus= product_by_date.Description.unique()

empty_dataframe= pd.DataFrame()

for sku in skus:
    a= product_by_date[product_by_date.Description== sku]
    a['previous_date']= a['date'].shift(1)
    empty_dataframe= pd.concat([empty_dataframe,a],axis=0)
    

empty_dataframe['Duration']= empty_dataframe['date']-empty_dataframe['previous_date']

empty_dataframe['duration']=empty_dataframe['Duration'].astype('string').str.replace('days 00:00:00.000000000','')

empty_dataframe['duration']= pd.to_numeric(empty_dataframe['duration'],errors='coerce')

ADI= empty_dataframe.groupby('Description').agg(ADI= ('duration','mean')).reset_index()

adi_cv= pd.merge(ADI, cv_data)


def category(dataframe):
    a=0
 
    if((dataframe['ADI']<= 1.34) & (dataframe['cv_squared']<= 0.49)):
      a='smooth'
    if((dataframe['ADI']>= 1.34) & (dataframe['cv_squared']>= 0.49)):
      a= 'Lumpy'
    if((dataframe['ADI']< 1.34) & (dataframe['cv_squared']> 0.49)):
      a='Erratic'
    if((dataframe['ADI']> 1.34) & (dataframe['cv_squared']< 0.49)):
      a='intermittent'
    return a


adi_cv['category']= adi_cv.apply(category, axis=1)
    

import seaborn as sns

sns.scatterplot(x='cv_squared',y='ADI',hue='category',data=adi_cv)



adi_cv[adi_cv.category==0]

adi_cv.category.value_counts()























