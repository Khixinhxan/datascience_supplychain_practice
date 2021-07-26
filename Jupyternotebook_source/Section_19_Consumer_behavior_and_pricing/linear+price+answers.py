#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:28:02 2020

@author: haythamomar
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

pineapple= pd.read_csv('pinapple_juice.csv')


X= pineapple[['Price']]
y= pineapple[['Pinapple juice']]

model= LinearRegression().fit(X,y)

intercept= model.intercept_[0]
coef= model.coef_[0]

pineapple.describe()

simulation_data= pd.DataFrame({'price': np.linspace(0.1,5,300)})

simulation_data['demand']= intercept+ simulation_data['price']*coef

simulation_data['cost']=simulation_data['demand'] *0.7

simulation_data['revenue']=simulation_data['demand'] *simulation_data['price']

simulation_data['profit']=simulation_data['revenue'] -simulation_data['cost']

import matplotlib.pyplot as plt

plt.plot(simulation_data['price'],simulation_data['revenue'],label='revenue')
plt.plot(simulation_data['price'],simulation_data['profit'],label='profit')
plt.legend(loc='upper right')

max_revenue=simulation_data[simulation_data['revenue']==max(simulation_data['revenue'])]

max_profit=simulation_data[simulation_data['profit']==max(simulation_data['profit'])]


# Section 19 – second assignment 
# In this section, I advise you to calculate manually 
# the elasticity of pineapple juice and then apply the
#  linear elasticity function on it and see if your 
#  calculations are correct. assume that your current price is 2.2 USD.


###Elasticity

# e= - (d(p)'*p)/d(p)

e= (-1* coef * 2.2)/(200 + 2.2*coef)

import inventorize as inv

inv.linear_elasticity(pineapple['Price'], 
                      pineapple['Pinapple juice'], 2.2, 0.7)





