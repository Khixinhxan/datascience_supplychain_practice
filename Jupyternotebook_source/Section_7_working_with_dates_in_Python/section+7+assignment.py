#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:12:38 2020

@author: haythamomar
"""
# Make a new script called Section 7 assignment.
#Import twentyeleven.csv

import pandas as pd
twentyeleven= pd.read_csv('twentyeleven.csv')

#	Drop duplicates if any from the dataset.

twentyeleven= twentyeleven.drop_duplicates()

#   get the week , to get the day of the week, the month and the year from invoice date column.

twentyeleven.info()
twentyeleven['InvoiceDate']=pd.to_datetime(twentyeleven['InvoiceDate'])
twentyeleven['week']= twentyeleven['InvoiceDate'].dt.week

twentyeleven['dayofweek']= twentyeleven['InvoiceDate'].dt.dayofweek

twentyeleven['month']=twentyeleven['InvoiceDate'].dt.month

twentyeleven['year']=twentyeleven['InvoiceDate'].dt.year

twentyeleven['week']=twentyeleven['InvoiceDate'].dt.week



#	Make a new column and called it month year with the month name and year.

twentyeleven['month_year']= twentyeleven['InvoiceDate'].dt.strftime('%B- %Y')


#	Get the last purchase date per customer
twentyeleven['date']= twentyeleven['InvoiceDate'].dt.strftime('%Y-%m-%d')

twentyeleven['date']=pd.to_datetime(twentyeleven['date'])

max_date= twentyeleven['date'].max()

customer_last= twentyeleven.groupby('Customer ID').agg(last_purchase_date= ('date','max')).reset_index()

customer_last['recency']= max_date-customer_last['last_purchase_date']


###changing recency to integer

customer_last['recency']=customer_last['recency'].astype('string').str.replace('days 00:00:00.000000000','')

customer_last['recency']= pd.to_numeric(customer_last['recency'],errors='coerce')

import matplotlib.pyplot as plt

plt.hist(customer_last['recency'])


#	get the recency per customer




#	apply two weeks and one week moving average  for sales on the data .

sales_per_day= twentyeleven.groupby('date').agg(total_sales=('Quantity','sum'))

sales_per_day.plot()

sales_per_day['moving_7']= sales_per_day.rolling(window=7).mean()
sales_per_day['moving_14']= sales_per_day.total_sales.rolling(window=14).mean()

sales_per_day['Aug-2011'].plot()






#	resample the data to weekly data using the sum of all observations on that week.

sales_resamples= sales_per_day.total_sales.resample('W').sum()



