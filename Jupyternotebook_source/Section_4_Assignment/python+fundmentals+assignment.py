#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:05:20 2020

@author: haythamomar
"""
# Welcome to our first assignment.


# Please open a new Python script and call it "section 4 assignment"
# in this assignment, we will work on the car's data set. it has the features of 400 cars from horsepower to speed and price.
# part 1:
    
import pandas as pd
import os 
path  = os.getcwd()
print(path)
cars=pd.read_csv('{0}/cars.csv'.format(path))   
    
cars.head()
cars.columns

# 1- How many Rows are in the cars dataset?


cars.shape[0]


# 2- How many Columns are in the car's data set?

cars.shape[1]



# 3- How many unique numbers of cylinders we have in the cars dataset?

cars.cylenders.value_counts()

cars.cylenders.unique()


# 4- what is the average horsepower of cars? 

cars.horsepower.describe()

# 5- what is the maximum horsepower?
500
# 6- what is the most expensive car?

cars[cars.Price== max(cars.Price)]

# 7- change the name of the column "name" to "car name"

cars= cars.rename(columns= {'name': 'car_name'})


# part 2:
# 8- make a subset of the data that has the car name, the price and name the new sub-setted data frame  car pricing.
cars_pricing= cars[['car_name','Price']]


# 9- create a function called pricing category that returns "Budget Car " if 
#the cars are less than 20,000 USD," Suitable Car " is the car 
#is more than 20,000 and less than 35 000 and 
#finally an expensive car for cars more than 35000. 

def pricing_category(price):
    if( price < 20000):
        a= 'Budget Car'
    if((price>= 20000)&(price<= 35000)):
        a= 'Suitable Car'
    if(price >35000):
        a= 'Expensive Car'
    return a

pricing_category(55000)





# 10- create a column named category on the subset using a for loop and pricing category function.

cars_pricing['category']='NA'

for i in range(cars_pricing.shape[0]):
    cars_pricing.category[i]= pricing_category(cars_pricing.Price[i])
    










# 11- How many Budget cars, suitable cars, and expensive cars we have?

cars_pricing.category.value_counts()


# As always, please first try to answer on your own and then have a look at the solved script(attached).

# All the best,
# Haytham
# Rescale analytics

