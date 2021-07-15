#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:28:22 2020

@author: haythamomar
"""
mean=50
sd=10
c=1.2
Salvage=0.7
penality_term=0.4
price=3
c=1.2
import inventorize as inv

inv.CriticalRatio(price, c, 0, 0)

(price-c)/(price-c+c)

inv.MPN_singleperiod(mean, sd, price, c, 0, 0)

inv.MPN_singleperiod(mean, sd, price, c, Salvage, 0)


inv.MPN_singleperiod(mean, sd, price, c, Salvage, penality_term)


inv.EPN_singleperiod(70, 40, 10, price, c, 0, 0)
import numpy as np
import pandas as pd
retail_clean= pd.read_csv('retail_clean.csv')

retail_clean.head()
retail_clean.columns

retail_clean.info()

retail_clean['InvoiceDate']= pd.to_datetime(retail_clean['InvoiceDate'])
retail_clean['date']= retail_clean['InvoiceDate'].dt.strftime('%Y-%m-%d')

retail_clean['date']=pd.to_datetime(retail_clean['date'])

retail_clean['year']=retail_clean['date'].dt.year

years_2= retail_clean[retail_clean.year.isin([2010,2011])]


total=years_2.groupby(['year','Description']).agg(total_sales=('Quantity',np.sum),
                                            price= ('Price','mean')).reset_index()


expected= total.groupby('Description').agg(expected_Demand= ('total_sales',np.mean),
                                           sd= ('total_sales','std'),
                                           price=('price',np.mean)).reset_index()
expected['sd']

def margin_error(dataframe):
    if(pd.isna(dataframe['sd'])):
        a= dataframe['expected_Demand']*0.1
    else:
        a= dataframe['sd']
    return a

expected['sd1']= expected.apply(margin_error,axis=1)    
expected['cost']= expected['price']*0.4


empty_data= pd.DataFrame()

for i in range(expected.shape[0]):
     a= inv.MPN_singleperiod(expected.loc[i,'expected_Demand'],
                             expected.loc[i,'sd1'],
                             expected.loc[i,'price'], expected.loc[i,'cost'], 0, 0)
     b= pd.DataFrame(a,index=[0])
     b['description']=expected.loc[i,'Description']
     empty_data= pd.concat([empty_data,b],axis=0)
     
empty_data     


empty_data.iloc[1,:]

     
     







































