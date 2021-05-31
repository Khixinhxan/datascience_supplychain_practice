#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 00:01:41 2020

@author: haythamomar
"""

import pandas as pd
import os 
path  = os.getcwd()
print(path)

flights= pd.read_csv('{0}/flights.csv'.format(path))
planes= pd.read_csv('{0}/planes.csv'.format(path))
airlines= pd.read_csv('{0}/airlines.csv'.format(path))
airports=pd.read_csv('{0}/airports.csv'.format(path))


flights.columns

flights.dep_time
flights.dep_delay

planes.head()
planes.columns

airlines

airports

##Open a script and name it to section 6 assignment and try to tackle the below questions:
    
#### what is the most popular destination city from NewYork?

flights.columns
table1=flights.groupby('dest').agg(count=('dest','count')).sort_values(by='count',ascending=False).reset_index()

pd.merge(table1, airports[['faa','name']],how='left',left_on='dest',right_on= 'faa')




### which month is the busiest of the year?

flights.columns
flights.month.value_counts()



#### which airline is the most punctual?

flights['total_delay']= flights['arr_delay']+flights['dep_delay']

table1=flights.groupby('carrier').agg(mean_delay= ('total_delay','mean')).sort_values(by='mean_delay').reset_index()
airlines
pd.merge(table1,airlines,how='left')








##### what destination has  the longest duration

flights.air_time.describe()

table1=flights.groupby(['origin','dest']).agg(average_air_time= 
                            ('air_time','mean')).reset_index().sort_values(by='average_air_time',ascending=False)
airports
pd.merge(table1,airports[['faa','name']],how='left',left_on= 'dest',right_on='faa')
#### what airline is the worst in terms of delays

##fronteir 


### which airline has the highest capacity of seats?

airlines

planes.columns
flights.columns

carrier_tailnum= flights[['carrier','tailnum']].drop_duplicates()

seats=pd.merge(carrier_tailnum,planes[['tailnum','seats']],how='left').groupby('carrier').agg(total_seats=('seats','sum')).sort_values(by='total_seats',ascending=False)



### which airplane model is the highest in use and from which manufacturer?


airplanes_use=flights.groupby('tailnum').agg(count= ('tailnum','count')).reset_index()

planes.columns
pd.merge(planes[['tailnum','model','manufacturer']],airplanes_use).groupby(['model','manufacturer']).agg(total_flights= ('count','sum')).sort_values(by='total_flights',ascending=False)



