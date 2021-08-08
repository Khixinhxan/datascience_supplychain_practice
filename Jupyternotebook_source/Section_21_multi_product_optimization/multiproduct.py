#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:35:51 2020

@author: haythamomar
"""
import pandas as pd
import inventorize as inv
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

multi= pd.read_csv('multvariate_slides.csv')

multi.iloc[:,1:5].describe()

X= multi.iloc[:,1:5]
X= sm.add_constant(X)
multi.columns
model_product1= sm.OLS(multi[['sales_product1']],X).fit()
model_product1.summary()

model_product2= sm.OLS(multi[['sales_product2']],X).fit()
model_product2.summary()

model_product3= sm.OLS(multi[['sales_product3']],X).fit()
model_product3.summary()


model_product4= sm.OLS(multi[['sales_product4']],X).fit()
model_product4.summary()


#### Multinomial Logit Models



import numpy as np
import numpy
import inventorize as inv
import pandas as pd
multi_choice=pd.read_csv('multi_slides.csv')

choices=inv.Multi_Competing_optimization(multi_choice.iloc[:4000,0:4],
                                         multi_choice.loc[:4000,'choice'],4, 
                                 [40,60,70,100])


choices_without_cost=inv.Multi_Competing_optimization(multi_choice.iloc[:4000,0:4],
                                         multi_choice.loc[:4000,'choice'],4, 
                                 [0,0,0,0])





