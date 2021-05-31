#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:16:15 2020

@author: dang
"""

### Statistical analysis assignment
from scipy.stats import norm, normaltest, kstest
import scipy.stats as st
import pandas as pd
import numpy as np

data = pd.read_csv('./Pinapple_juice.csv')

# Fit the demand of this Distribution to normal demand

demand = data['Pinapple juice']

mean = demand.mean()
std = demand.std()

norm_test = kstest(demand, 'norm', args=(mean,std))

#statistic
norm_test[0]
#pvalue
norm_test[1]

result = []
parameters = {}

norm_param = getattr(st, 'norm')

norm_param.fit(demand)

dist_names = ["norm", "exponweib", "weibull_max", 
              "weibull_min", "pareto", "genextreme"]

for dist in dist_names:
    param = getattr(st, dist)
    fitting = param.fit(demand)
    test = kstest(demand, dist, args=fitting)
    result.append([dist, test])
    print("the result for dist {0} is test {1}".format(dist, str(test[1])))

### 4 Make a linear regression using LM function y~x 
###and outline the coeffients and the intercept

from sklearn.linear_model import LinearRegression

data.columns

data.rename(columns={"Pinapple juice": "Pinapple_juice"}, inplace=True)
X = data.Pinapple_juice.values.reshape(1,-1)
y = data.Price

data[['Pinapple_juice', 'Price']].corr

model = LinearRegression()
model.fit(X, y)
