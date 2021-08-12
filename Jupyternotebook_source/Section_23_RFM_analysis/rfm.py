#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:57:57 2020

@author: haythamomar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


retail= pd.read_csv('retail_clean.csv')

retail['InvoiceDate']= pd.to_datetime(retail['InvoiceDate'])
retail['date']= retail['InvoiceDate'].dt.strftime('%Y-%m-%d')
retail['date']= pd.to_datetime(retail['date'])

max_date= retail['date'].max()

retail.columns
customers_recency= retail.groupby('Customer ID').agg(last_date= ('date','max')).reset_index()
customers_recency['recency']= max_date- customers_recency['last_date']
customers_recency['recency']=customers_recency['recency'].astype('string').str.replace('days','').astype(int)

#### frequency

freq2= retail.groupby('Customer ID').date.count().reset_index()
freq2.columns= ['Customer ID','frequency']

freq2






### Monetary value
retail.columns

monet1= retail.groupby(['Customer ID','Invoice']).agg(revenue= ('Revenue','sum')).reset_index()

monet2= monet1.groupby('Customer ID').agg(monetary= ('revenue','mean')).reset_index()


customers_recency['rank_recency']= customers_recency['recency'].rank(pct=True)
freq2['freq_ranking']=freq2['frequency'].rank(ascending=False,pct=True)


monet2['rank_monet']= monet2['monetary'].rank(ascending=False,pct=True)

all_data= pd.merge(customers_recency, freq2,how='left',on='Customer ID')
all_data= pd.merge(all_data,monet2,how='left',on='Customer ID')

bins=[0,0.5,1]
names= ['1','2']

final= pd.DataFrame(customers_recency['Customer ID'])

final['frequency']= pd.cut(freq2['freq_ranking'],bins,labels=names).astype('string')
final['recency']= pd.cut(customers_recency['rank_recency'],bins,labels=names).astype('string')
final['monetary']=pd.cut(monet2['rank_monet'],bins,labels=names).astype('string')

final['rec_freq_mone']=final['recency']+final['frequency']+final['monetary']

all_data['rec_freq_monet']= final['rec_freq_mone']

all_data.to_csv('rfm.csv')
import seaborn as sns

fig= sns.countplot(x='rec_freq_monet',data= all_data)

fig.set_xticklabels(fig.get_xticklabels(),
                   rotation=45 )































