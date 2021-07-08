#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:40:49 2020

@author: haythamomar
"""

import pandas as pd
import inventorize3 as inv
import datetime
import numpy as np
retail= pd.read_csv('online_retail2.csv')

retail= retail.drop_duplicates()
retail= retail.dropna(axis=0)


retail.info()

retail.InvoiceDate
retail['InvoiceDate']= pd.to_datetime(retail['InvoiceDate'])

retail['date']= retail.InvoiceDate.dt.strftime('%Y-%m-%d')

retail['date']=pd.to_datetime(retail['date'])

max_data= max(retail.date)

last_four= retail[retail.date > '2011-08-01']
last_four.columns

last_four['revenue']= last_four['Quantity']* last_four['Price']


a=last_four.groupby(['date','Description']).agg(total_daily= ('Quantity',np.sum),
                                             total_revenue= ('revenue',np.sum)).reset_index()

grouped= a.groupby('Description').agg(average= ('total_daily',np.mean),
                                      sd= ('total_daily','std'),
                                      total_sales= ('total_daily',np.sum),
                                      total_revenue=('total_revenue',np.sum)).reset_index()

for_abc= inv.productmix(grouped['Description'], grouped['total_sales'], grouped['total_revenue'])


for_abc.product_mix.value_counts()

lead_time=12
sd_leadtime=2

mapping={'A_A':0.95,"A_C": 0.95,"C_A":0.8,"A_B":0.95,
         'B_A':0.7,"B_C":0.75,"C_C":0.7,"B_B":0.7,"C_B": 0.8}

for_abc['service_level']=for_abc.product_mix.map(mapping)


### reorder point 

pd.DataFrame(inv.reorderpoint(80, 10, 12, 0.75),index=[0])


abcd= for_abc[['skus','service_level']]


for_reorder=pd.merge(grouped, abcd,how='left',left_on='Description',right_on='skus')
for_reorder.columns

empty_data= pd.DataFrame()
for i in range(for_reorder.shape[0]):
    ordering_point= inv.reorderpoint(for_reorder.loc[i,'average'],
                                     for_reorder.loc[i,'sd'],
                                     12, for_reorder.loc[i,'service_level'])
    as_data= pd.DataFrame(ordering_point,index=[0])
    as_data['Description']= for_reorder.loc[i,'Description']
    empty_data=pd.concat([empty_data,as_data],axis=0)
    
    
empty_data 

### joinning all
all_data= pd.merge(for_reorder, empty_data,how='left')

all_data.columns


all_data['saftey_stock']=all_data['reorder_point']-all_data['demandleadtime']

all_data[all_data.saftey_stock== max(all_data.saftey_stock)]
import seaborn as sns

all_data=all_data[all_data.saftey_stock != max(all_data.saftey_stock)]
sns.scatterplot(x='sd',y='saftey_stock',hue='service_level',data=all_data)



#### with leadtime variability 

empty_data_ltv= pd.DataFrame()
for i in range(for_reorder.shape[0]):
    ordering_point= inv.reorderpoint_leadtime_variability(int(for_reorder.loc[i,'average']),
                                     for_reorder.loc[i,'sd'],
                                     12,2, for_reorder.loc[i,'service_level'])
    as_data= pd.DataFrame(ordering_point,index=[0])
    as_data['Description']= for_reorder.loc[i,'Description']
    empty_data_ltv=pd.concat([empty_data_ltv,as_data],axis=0)
    
    
empty_data_ltv 

### joinning all
all_data_ltv= pd.merge(for_reorder, empty_data_ltv,how='left')

all_data.columns












all_data['saftey_stock']=all_data['reorder_point']-all_data['demandleadtime']

all_data[all_data.saftey_stock== max(all_data.saftey_stock)]
import seaborn as sns

all_data=all_data[all_data.saftey_stock != max(all_data.saftey_stock)]
sns.scatterplot(x='sd',y='saftey_stock',hue='service_level',data=all_data)


    
    
    
    




























