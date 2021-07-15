#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:42:33 2020

@author: haythamomar
"""
import pandas as pd
import inventorize as inv
import numpy as np
import array
skus= pd.read_csv('sku_distributions.csv')


apple_juice= skus[['apple_juice']]

mean_apple= apple_juice.mean()

sd_apple= apple_juice.std()


leadtime=7 


apple_sq=inv.sim_min_Q_normal(apple_juice, mean_apple, sd_apple, leadtime=7,
                     service_level=0.8, Quantity=100,shortage_cost=1,ordering_cost=1,inventory_cost=1)


apple_sq[0].to_csv('Apple_sq.csv')

apple_sq1=inv.sim_min_Q_normal(apple_juice, mean_apple, sd_apple, leadtime=7,
                     service_level=0.8, Quantity=100,shortage_cost=1,ordering_cost=1,inventory_cost=1)


apple_sq2=inv.sim_min_Q_normal(apple_juice, mean_apple, sd_apple, leadtime=7,
                     service_level=0.9, Quantity=300,shortage_cost=1,ordering_cost=1,inventory_cost=1)

skus

grape_juice= inv.sim_min_max_pois(skus.grape_juice, lambda1=skus.grape_juice.mean(), 
                                  leadtime=7, service_level=0.8, Max= 30,
                                  shortage_cost=1,ordering_cost=1,inventory_cost=1)

grape_juice[0].to_csv('grape_juice.csv')


skus

cantalop_juice= skus[['cantalop_juice']]


cantalop=inv.Periodic_review_pois(cantalop_juice, lambda1= cantalop_juice.mean(),
                         leadtime=7, service_level=0.9, Review_period=3,ordering_cost=1,
                         inventory_cost=1,shortage_cost=1)



cantalop_hybrid= inv.Hibrid_pois(cantalop_juice, lambda1= cantalop_juice.mean(),leadtime=7,
 service_level=0.9, Review_period=3,
                         inventory_cost=1,shortage_cost=1,Min=120, ordering_cost=1
                         )


cantalop_hybrid[0].to_csv('cantalop_hybrid.csv')


###base policy



apple_sq=inv.sim_min_Q_normal(apple_juice, mean_apple, sd_apple, leadtime=7,
                     service_level=0.8, Quantity=400,shortage_cost=1,ordering_cost=100,inventory_cost=1)


apple_base= inv.sim_base_normal(apple_juice, mean_apple, sd_apple, leadtime=7,
                     service_level=0.8,shortage_cost=1,ordering_cost=100,inventory_cost=1)




#####comparison

apple_sq=inv.sim_min_Q_normal(apple_juice, mean_apple, sd_apple, leadtime=7,
                     service_level=0.8, Quantity=100,shortage_cost=1,ordering_cost=100,inventory_cost=1)


apple_base= inv.sim_base_normal(apple_juice, mean_apple, sd_apple, leadtime=2,
                     service_level=0.8,shortage_cost=1,ordering_cost=100,inventory_cost=1)

appleminmax=inv.sim_min_max_normal(apple_juice, mean_apple, sd_apple, leadtime=2,Max=400,
                     service_level=0.8,shortage_cost=1,ordering_cost=100,inventory_cost=1)

apple_periodic=inv.Periodic_review_normal(apple_juice, mean_apple, sd_apple, leadtime=2,
                     service_level=0.8,shortage_cost=1,ordering_cost=100,inventory_cost=1,Review_period=4)


apple_hibrid=inv.Hibrid_normal(apple_juice, mean_apple, sd_apple, leadtime=2,Min=200,
                     service_level=0.8,shortage_cost=1,ordering_cost=100,inventory_cost=1,Review_period=3)

import matplotlib.pyplot as plt

plt.subplot(2,2,1)

plt.plot(apple_sq[0].period[5:],apple_sq[0].demand[5:],label='demand')
plt.plot(apple_sq[0].period[5:],apple_sq[0].sales[5:],label='sales')
plt.plot(apple_sq[0].period[5:],apple_sq[0].order[5:],label='order')
plt.scatter(apple_sq[0].period[5:],apple_sq[0].inventory_level[5:],label='inventory')
plt.title('MINQ')
plt.legend(loc='upperleft')

plt.subplot(2,2,2)

plt.plot(apple_base[0].period[5:],apple_base[0].demand[5:],label='demand')
plt.plot(apple_base[0].period[5:],apple_base[0].sales[5:],label='sales')
plt.plot(apple_base[0].period[5:],apple_base[0].order[5:],label='order')
plt.scatter(apple_base[0].period[5:],apple_base[0].inventory_level[5:],label='inventory')
plt.title('base')
plt.legend(loc='upperleft')

plt.subplot(2,2,3)

plt.plot(appleminmax[0].period[5:],appleminmax[0].demand[5:],label='demand')
plt.plot(appleminmax[0].period[5:],appleminmax[0].sales[5:],label='sales')
plt.plot(appleminmax[0].period[5:],appleminmax[0].order[5:],label='order')
plt.scatter(appleminmax[0].period[5:],appleminmax[0].inventory_level[5:],label='inventory')
plt.title('MinMax')
plt.legend(loc='upperleft')

plt.subplot(2,2,4)

plt.plot(apple_periodic[0].period[5:],apple_periodic[0].demand[5:],label='demand')
plt.plot(apple_periodic[0].period[5:],apple_periodic[0].sales[5:],label='sales')
plt.plot(apple_periodic[0].period[5:],apple_periodic[0].order[5:],label='order')
plt.scatter(apple_periodic[0].period[5:],apple_periodic[0].inventory_level[5:],label='inventory')
plt.title('periodic')
plt.legend(loc='upperleft')

plt.show()







