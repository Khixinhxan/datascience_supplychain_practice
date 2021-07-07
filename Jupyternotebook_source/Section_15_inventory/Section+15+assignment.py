#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:07:57 2020

@author: haythamomar
"""
# In this exercise, we will solve this problem by using python  exactly
#  as per the previous lecture.
# you are the supply chain manager of coffee co, a well-known 
# brand for coffee Columbian blend
# coffee co takes raw coffee from its partner supplier in 
# Columbia. the annual demand for raw coffee
# is 6000 tons per year, the price of one ton from the 
# supplier is 1500 USD, assume a holding
# #rate of 10% while the cost of transportation and 
# order is 4000 USD. what should be the optimal Q.
# what is the total logistics cost? what is your t 
# practical? if the supplier will offer you a 10 % discount if your 
# Q is 700, would you accept it?
# if the Lead time from the Columbian supplier is 2
#  months, what is the reorder point?


import inventorize as inv
import pandas as pd
d= 6000
c= 1500
s= 4000
h=0.1

eoq= pd.DataFrame(inv.eoq(d, s, c, h),index=[0])

eoq1= eoq.loc[0,'EOQ']


### d/q * s +  q/2 *(h*p) + D* p

TLC= (d/eoq1) *s + (eoq1/2)*(h*c)+ c*d

inv.TQpractical(d, s, c, h)



### at 10% discount

TLC1= (d/700) *s + (700/2)*(h*c*0.9)+ c*0.9*d


### welcome . the lead time it takes for the orders to arrive is two month, what will be the
#reorder-point:
    
t= eoq1/d    

L= 2/12

L> t

l_prime= L-(1*(eoq1/d))
reorder_point= l_prime*d

# Whenever our invenotry level /Inventory position , we order Q and Q in this case is 438


### welcome . the lead time it takes for the orders to arrive is two month, what will be the
#reorder-point:
    
L2=2/12
L2<t

l_prime= L2- (1* (eoq1/d))


reorder_point= l_prime *d





