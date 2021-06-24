#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:28:54 2020

@author: haythamomar
"""
### pip install pulp

from pulp import *

product1= 25
product2=35

## 25 kg of feather
## 35 for cotton

## it takes 0.3 k f and 0.5 k c to make x1
## it takes 0.5 k f and 0.5 k c to make x2

model = LpProblem('PILLOWS',LpMaximize)

X1= LpVariable('X1',0,None,'Integer')
X2= LpVariable('X2',0,None,'Integer')

##define our objeective function

model += X1 *25 +X2 *35

model += X1* 0.3 +X2 * 0.5 <= 20
model += X1* 0.5 +X2 * 0.5 <= 35

model.solve()

X1.varValue
X2.varValue


from pulp import *

model= LpProblem('shipping',LpMinimize)

customers=['Australia','Sweeden','Brazil']
factory= ['Factory1','Factory2']
products= ['Chair','Table','Beds']

keys= [(f,p,c) for f in factory for p in products for c in customers]

var= LpVariable.dicts('shipment', keys,0,None,cat='Integer')



costs_value= [50,80,50,
        60,90,60,
        70,90,70,
        80,50,80,
        90,60,90,
        90,70,90]

costs= dict(zip(keys,costs_value))




demand_keys= [(p,c)for c in customers
              for p in products]
demand_values=[50,80,200,
               120,80,40,
               30,60,175]
demand= dict(zip(demand_keys,demand_values))


model+= lpSum(var[(f,p,c)]*costs[(f,p,c)]
   for f in factory for p in products for c in customers )

model += lpSum(var[('Factory1',p,c)]
               for p in products for c in customers)<= 500
model += lpSum(var[('Factory2',p,c)]
               for p in products for c in customers)<= 500

for c in customers:
    for p in products:
        model += var[('Factory1',p,c)]+var[('Factory2',p,c)]>= demand[(p,c)]

model.solve()

for i in var: 
    print('{} shipping {}'.format(i,var[i].varValue))










import pandas as pd
import pulp as *
DC_model= LpProblem('DC',LpMinimize)



####Demand
demand= pd.DataFrame({'Australia': [50000,30000,45000],
                      'Sweeden': [12000,80000,40000],
                     'Brazil': [30000,60000,175000]},
                      index= ['Chair','Table','Beds'])
 

### Paremeter names
customers=['Australia','Sweeden','Brazil']
factory= ['Factory1','Factory2']
products= ['Chair','Table','Beds']
warehouse= ['DC1','DC2']

##production
production_keys= [ (f,p) for f in factory for p in products]

P_integ= LpVariable.dicts('prod_', production_keys,0,None,cat='Integer')
P_bin=LpVariable.dicts('prod_Fixed', production_keys,cat='Binary')
P_var=[50,60,70,80,90,90]
P_fix=[30000,25000,50000,25000,40000,40000]
capacity= [100000,100000,130000,100000,100000,130000]
capacity_p=dict(zip(production_keys,capacity))
P_costs= dict(zip(production_keys,P_var))
P_open= dict(zip(production_keys,P_fix))


inbound_keys= [(f,p,w) for f in factory for p in products for w in warehouse]

inbound_cost= [10,4,20,5,5,6,
               2,10,3,12,4,15]
inbound_c= dict(zip(inbound_keys,inbound_cost))

inbound_var= LpVariable.dicts('in', inbound_keys,0,None,cat='Integer')

outbound= [(w,p,c)for w in warehouse
           for p in products
           for c in customers]
outbound_cost=[8,7,8,
               9,6,9,10,12,10,
               7,4,10,
               6,5,12,12,6,15]
outbound_c= dict(zip(outbound,
                     outbound_cost))
outbound_var= LpVariable.dicts('out', outbound,0,None,'Continous')


DC_model += (lpSum(P_integ[(f,p)]* P_costs[(f,p)]for p in products for f in factory)
+ lpSum(P_bin[(f,p)]* P_open[(f,p)]for p in products for f in factory)
+ lpSum(inbound_var[(f,p,w)]* inbound_c[(f,p,w)]for p in products for f in factory for w in warehouse)
+ lpSum(outbound_var[(w,p,c)]* outbound_c[(w,p,c)]for p in products for w in warehouse for w in warehouse))

### supply constraints

for p in products:
    for f in factory:
        DC_model += P_integ[(f,p)] >= lpSum(capacity_p[(f,p)]*P_bin[(f,p)])

###inbound
for f in factory :
    for p in products:
         DC_model += P_integ[(f,p)] == inbound_var[(f,p,'DC1')]+ inbound_var[(f,p,'DC2')]

### outbound 

for p in products :
    for w in warehouse :
        DC_model += (inbound_var[('Factory1',p,w)]+inbound_var
        [('Factory2',p,w)])== (outbound_var[(w,p,'Australia')]+
                              outbound_var[(w,p,'Brazil')]+outbound_var[(w,p,'Sweeden')])

### Demand Constraints:
    for p in products:
        for c in customers:
            DC_model+= (outbound_var[('DC1',p,c)]+outbound_var[('DC2',p,c)]) >= demand.loc[p,c]




import pandas as pd
from pulp import *

param=pd.read_excel('Production_scheduling.xlsx')


param=param.rename(columns={'Unnamed: 0': 'period'} )
param['Capacity']=5000
param['t']= range(1,13)

param= param.set_index('t')

inventory= LpVariable.dicts('inv',[0,1,2,3,4,5,6,7,8,9,10,11,12],0,None,'Integer')
inventory[0]= 200

production=LpVariable.dicts('Prod',[1,2,3,4,5,6,7,8,9,10,11,12],0,None,'Integer')
binary= LpVariable.dicts('binary',[1,2,3,4,5,6,7,8,9,10,11,12],0,None,'Binary')

time= [1,2,3,4,5,6,7,8,9,10,11,12]


model= LpProblem('Production',LpMinimize)

model += lpSum([ inventory[t]* param.loc[t,'storage cost']+ production[t]* param.loc[t,'var']+
                binary[t]* param.loc[t,'fixed cost'] for t in time])


for t in time:
    model+=  production[t]  -  inventory[t]+ inventory[t-1]>= param.loc[t,'demand']
    model +=   production[t]<=        binary[t]* param.loc[t,'Capacity']
    
model.solve()    
for v in model.variables():
    print(v,v.varValue)

for i in production: print(production[i],production[i].varValue)



    
    
    































