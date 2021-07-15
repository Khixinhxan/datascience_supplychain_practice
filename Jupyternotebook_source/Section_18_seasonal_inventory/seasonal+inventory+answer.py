#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:48:06 2020

@author: haythamomar
"""

import pandas as pd
import inventorize as inv


retail= pd.read_csv('twentyeleven.csv')



retail['InvoiceDate']= pd.to_datetime(retail['InvoiceDate'])
retail['date']= retail['InvoiceDate'].dt.strftime('%Y-%m-%d')

retail['date']=pd.to_datetime(retail['date'])

retail['year']=retail['date'].dt.year


total=retail.groupby(['Description']).agg(total_sales=('Quantity','sum'),
                                            price= ('Price','mean')).reset_index()


total['sd']= total['total_sales']*0.1

total['cost']= total['price']*0.4



empty_data= pd.DataFrame()

for i in range(total.shape[0]):
     a= inv.MPN_singleperiod(total.loc[i,'total_sales'],
                             total.loc[i,'sd'],
                             total.loc[i,'price'], total.loc[i,'cost'], 0, 0)
     b= pd.DataFrame(a,index=[0])
     b['description']=total.loc[i,'Description']
     empty_data= pd.concat([empty_data,b],axis=0)
     
empty_data     


empty_data.iloc[1,:]

   







