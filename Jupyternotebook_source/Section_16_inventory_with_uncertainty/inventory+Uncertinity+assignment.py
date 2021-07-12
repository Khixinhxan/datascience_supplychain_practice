#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:24:57 2020

@author: haythamomar
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:40:49 2020

@author: haythamomar
"""

import pandas as pd
import inventorize as inv
import datetime
import numpy as np
retail= pd.read_csv('UK_ireland.csv')

retail= retail.drop_duplicates()
retail= retail.dropna(axis=0)


retail.info()

retail.InvoiceDate
retail['InvoiceDate']= pd.to_datetime(retail['InvoiceDate'])

retail['date']= retail.InvoiceDate.dt.strftime('%Y-%m-%d')

retail['date']=pd.to_datetime(retail['date'])



max_data= max(retail.date)

last_three= retail[retail.date > '2011-09-09']
last_three.columns

last_three['revenue']= last_three['Quantity']* last_three['Price']


a=last_three.groupby(['date','Description']).agg(total_daily= ('Quantity',np.sum),
                                             total_revenue= ('revenue',np.sum)).reset_index()

grouped= a.groupby('Description').agg(average= ('total_daily',np.mean),
                                      sd= ('total_daily','std'),
                                      total_sales= ('total_daily',np.sum),
                                      total_revenue=('total_revenue',np.sum)).reset_index()

for_abc= inv.productmix(grouped['Description'], grouped['total_sales'], grouped['total_revenue'])


for_abc.product_mix.value_counts()

lead_time=21
sd_leadtime=2

mapping={'A_A':0.8,"A_C": 0.70,"C_A":0.8,"A_B":0.80,
         'B_A':0.8,"B_C":0.6,"C_C":0.6,"B_B":0.7,"C_B": 0.6}

for_abc['service_level']=for_abc.product_mix.map(mapping)


### reorder point 



abcd= for_abc[['skus','service_level']]


for_reorder=pd.merge(grouped, abcd,how='left',left_on='Description',right_on='skus')
for_reorder.columns




#### with leadtime variability 

empty_data_ltv= pd.DataFrame()
for i in range(for_reorder.shape[0]):
    ordering_point= inv.reorderpoint_leadtime_variability(int(for_reorder.loc[i,'average']),
                                     for_reorder.loc[i,'sd'],
                                     21,2, for_reorder.loc[i,'service_level'])
    as_data= pd.DataFrame(ordering_point,index=[0])
    as_data['Description']= for_reorder.loc[i,'Description']
    empty_data_ltv=pd.concat([empty_data_ltv,as_data],axis=0)
    
    
empty_data_ltv 

### joinning all
all_data= pd.merge(for_reorder, empty_data_ltv,how='left')


all_data['saftey_stock']=all_data['reorder_point']-all_data['demandleadtime']

all_data[all_data.saftey_stock== max(all_data.saftey_stock)]
import seaborn as sns

all_data=all_data[all_data.saftey_stock != max(all_data.saftey_stock)]
sns.scatterplot(x='sd',y='saftey_stock',hue='service_level',data=all_data)



















all_data['saftey_stock']=all_data['reorder_point']-all_data['demandleadtime']

all_data[all_data.saftey_stock== max(all_data.saftey_stock)]
import seaborn as sns

all_data=all_data[all_data.saftey_stock != max(all_data.saftey_stock)]
sns.scatterplot(x='sd',y='saftey_stock',hue='service_level',data=all_data)


    
    
    
    




























