#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:54:56 2020

@author: haythamomar
"""


# #- make a new script and call it the section 8 assignment.

# #Please try to answer the questions first and then have a look at the solved script.






# #2- import twentyeleven.csv ,iris, cars and the requested packages.

import pandas as pd
import seaborn as sns

twenty= pd.read_csv('twentyeleven.csv')
cars= pd.read_csv('cars.csv')
iris=pd.read_csv('iris.csv')

twenty.info()
twenty['InvoiceDate']=pd.to_datetime(twenty['InvoiceDate'])
twenty['date']= twenty['InvoiceDate'].dt.strftime('%Y-%m-%d')
twenty['date']= pd.to_datetime(twenty['date'])



# 3- Make a line plot of the sales of 2011 for the united kingdom.

uk= twenty[twenty.Country == 'United Kingdom']


sales_per_day= uk.groupby('date').agg(total_sales=('Quantity','sum'))

sales_per_day.plot()




# 4- for the next plot; select country countries<-c("Canada","Denmark","EIRE","United Kingdom")
#  make a line plot per each country using  plt subplots

countries=["Canada","Denmark","EIRE","United Kingdom"]
four_countries= twenty[twenty.Country.isin(countries)]

sales_per_Day= four_countries.groupby(['Country','date']).agg(total_sales=('Quantity','sum')).reset_index()

sales_pivoted= pd.pivot_table(sales_per_Day,values= 'total_sales',
                              columns='Country',index='date',fill_value=0)

sales_pivoted.plot(subplots=True)





# 5- Make a scatter plot for cars between price and horsepower.

sns.scatterplot(x= 'Price',y='horsepower',data=cars)



# 6- Make a distribution plot of sepal length in iris and segregate it by flower.

setosa= iris[iris.species=='setosa']
virginica= iris[iris.species== 'virginica']
versicolor=iris[iris.species== 'versicolor']

fig=sns.kdeplot(setosa.sepal_length,label='setosa')
fig=sns.kdeplot(virginica.sepal_length,label='virginica')
fig=sns.kdeplot(versicolor.sepal_length,label='versicolor')



# 7- Make a boxplot for the number of cylinders of cars, make sure to take only 4,6 
#and eight cylinders.

common_cylenders= cars[cars.cylenders.isin([4,6,8])]

sns.boxplot(x='cylenders',y='horsepower',data=common_cylenders)



# 8- make a pairplot of iris dataset segregated by flower type.

sns.pairplot(iris,hue='species')






