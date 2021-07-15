#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 11:08:57 2020

@author: haythamomar
"""
import pandas as pd
import inventorize as inv

pineapple= pd.read_csv('pinapple_juice.csv')

demand= pineapple['Pinapple juice']
leadtime=7
mean= pineapple['Pinapple juice'].mean()
sd= pineapple['Pinapple juice'].std()


#####comparison

pine_sq=inv.sim_min_Q_normal(demand, mean, sd, leadtime=7,
                     service_level=0.8, Quantity=1000,ordering_cost=100,
                     inventory_cost=5)


pine_base= inv.sim_base_normal(demand, mean, sd, leadtime=7,
                     service_level=0.8,ordering_cost=100,inventory_cost=5)

pine_minmax=inv.sim_min_max_normal(demand, mean, sd, leadtime=7,Max=1100,
                     service_level=0.8,ordering_cost=100,inventory_cost=5)

pine_periodic=inv.Periodic_review_normal(demand, mean, sd, leadtime=7,
                     service_level=0.8,ordering_cost=100,inventory_cost=5,
                     Review_period=7)


pine_hibrid=inv.Hibrid_normal(demand, mean, sd, leadtime=7,Min=20,
                     service_level=0.8,ordering_cost=100,inventory_cost=5,
                     Review_period=7)


pine_sq[1]
pine_base[1]
pine_minmax[1]
pine_periodic[1]
pine_hibrid[1]


