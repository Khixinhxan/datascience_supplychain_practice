#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 10:34:14 2020

@author: haythamomar
"""
#you are the supply chain manager of coffee co , a well known brand for 
#coffee columbian blend
#coffee co takes raw coffee from it's partner supplier in Columbia . 
#the annual demand of raw coffee
#is 4000 tons per year , the price of one ton from the supplier 
#is 2500 USD , assume a holding
#rate of 10% while the cost of transportation and ordering is
#6000 USD. what should be the optimal Q.
#what is the total logistics cost ? what is your t practical?
#if the supplier will offer you a 10 % discount if your Q is
#500, would you accept it.


import inventorize3 as inv
import pandas as pd
d= 4000
c= 2500
s= 6000
h=0.1

eoq= pd.DataFrame(inv.eoq(d, s, c, h),index=[0])

eoq1= eoq.loc[0,'EOQ']


### d/q * s +  q/2 *(h*p) + D* p

TLC= (d/eoq1) *s + (eoq1/2)*(h*c)+ c*d

inv.TQpractical(d, s, c, h)



### at 10% discount

TLC1= (d/500) *s + (500/2)*(h*c*0.9)+ c*0.9*d


### welcome . the lead time it takes for the orders to arrive is one month, what will be the
#reorder-point:
    
t= eoq1/d    

L= 1/12

L< t

reorderpoint = L *d

# Whenever our invenotry level /Inventory position , we order Q and Q in this case is 438


### welcome . the lead time it takes for the orders to arrive is two month, what will be the
#reorder-point:
    
L2=2/12
L2<t

l_prime= L2- (1* (eoq1/d))


reorder_point= l_prime *d



## MIN Q 



    
    
    

