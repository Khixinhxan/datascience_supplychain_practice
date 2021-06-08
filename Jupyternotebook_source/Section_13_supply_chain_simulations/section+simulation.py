#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:44:20 2020

@author: haythamomar
"""
import numpy as np

lambda1= 1
mean_service=1
sd= 0.2

arrival_time= np.random.exponential(1,400).cumsum()

service_time= np.random.normal(1,0.2,400)

def waiting_mean(arrival_time,service_time):
    waiting_time=[]
    leaving_time=[]
    
    waiting_time.append(0)
    leaving_time.append(arrival_time[0]+service_time[0]+waiting_time[0])
    
    for i in range(1,len(arrival_time)):
        waiting_time.append(max(0,leaving_time[i-1]-arrival_time[i]))
        leaving_time.append(arrival_time[i]+service_time[i]+waiting_time[i])
        
    mean_waiting= np.mean(waiting_time)  
    return mean_waiting

waiting_mean(arrival_time,service_time)

average_Sim= []
for i in range(0,1000):
    arrival_time= np.random.exponential(1,400).cumsum()

    service_time= np.random.normal(1,0.2,400)
    waiting_time= waiting_mean(arrival_time, service_time)
    average_Sim.append(waiting_time)
    
np.mean(average_Sim) 
np.median(average_Sim)

import matplotlib.pyplot as plt

plt.hist(average_Sim)  
    

from rpy2.robjects import r ,pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector 

pandas2ri.activate()



importr('queuecomputer') 


####queue_step(arrival_time,service_time,number_of servers)
   
import numpy as np
arrival_time= np.random.exponential(1,400).cumsum()

service_time= np.random.normal(1,0.2,400)

r_arrival= FloatVector(arrival_time)
r_service_time= FloatVector(service_time)

simulation= r['queue_step'](r_arrival,r_service_time,1)

simulation[0]
simulation[1]
simulation[2]
simulation[3]

mean_waiting_time=simulation[2]['waiting'].mean()

average_sim_r=[]
for i in range(0,1000):
    arrival_time= np.random.exponential(1,400).cumsum()

    service_time= np.random.normal(1,0.2,400)

    r_arrival= FloatVector(arrival_time)
    r_service_time= FloatVector(service_time)
    simulation=r['queue_step'](r_arrival,r_service_time,1)
    average_sim_r.append(simulation[2]['waiting'].mean())

import seaborn as sns
sns.distplot(average_sim_r)
np.median(average_sim_r)


##Problem 2: you are the operations manager at a call center , you have
#been assigned the task of determining how many call center rep you will
#need on an hourly basis , you checked the incoming call of customers one day
##and you have determined that you receive around 40 calls per hour , and a call
#takes around 7 minutes. How many
#call center reps you should have to have a waiting 
#time of no more than 5 minutes.
#that the arrival is exponential so as the service time.

import numpy as np
import random
from rpy2.robjects import r ,pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector 
pandas2ri.activate()
importr('queuecomputer') 
import seaborn as sns

rate= 40/60

arrival_time= np.cumsum([random.expovariate(rate) for x in range(1000)])
inter_arrival_time=[random.expovariate(rate) for x in range(1000)]
sns.distplot(inter_arrival_time)

service_time= np.random.exponential(7,1000)
sns.distplot(service_time)

r_arrival= FloatVector(arrival_time)
r_service= FloatVector(service_time)

simulation=r['queue_step'](r_arrival,r_service,5)

simulation[2]['waiting'].mean()


n_servers= range(1,9)
waiting_list=[]
for k in  n_servers:
    simulation=r['queue_step'](r_arrival,r_service,k)
    waiting_time=simulation[2]['waiting'].mean()
    waiting_list.append(waiting_time)
    
import matplotlib.pyplot as plt

plt.plot(n_servers,waiting_list)

list(zip(n_servers,waiting_list))



print(r['summary'](simulation))



### Problem 2 :

#now let's say you are a bank operations manager , and you would like to know 
#how many 
#tellers you need if your bank is visited by around 150 customers per 
#hour and the average
#serving time is exponentially distributed we a Mu of 15 minutes .
#note that the capacity of the
#bank is only 55 customers inside the bank.
#M,M,K,55,inf, you want the 
#maximum waiting time to be 10 min per customer until She/he is served.

import numpy as np
import random
from rpy2.robjects import r ,pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector 
pandas2ri.activate()
importr('queuecomputer') 
import seaborn as sns

arrival_Rate= 150/60
service_mean=15

arrival_time= np.cumsum([random.expovariate(arrival_Rate) for x in range(2000)])

service_time= np.random.exponential(15,2000)

r_arrival=FloatVector(arrival_time)
r_service_time= FloatVector(service_time)

simulation= r['queue_step'](r_arrival,r_service_time,38)

simulation[2]['waiting'].mean()
simulation[3]['queuelength'].max()



#####Multiple service
### at the same bank you made a further analysis , and you found out that
## the bank actually has two services , the teller service and the customer
## service support .the customers come the bank to register for either
# a teller or a customer service support , you have found out a customer
### takes around 30 seconds to register and 65% of the customers go to the tellers
### while 35% of the customers got to the customer service support, customers
#still arrive at an exponential rate of 150 per hour , the teller service takes
#10 minutes with a standard deviation of two while the customer service takes
## an exponential rate of 13 minutes. what is the mean waiting time of the system
#if you have 15 tellers and 15 customer service specialists.

arrival_registration= np.cumsum([random.expovariate(150/60) for x in range(5000)])
service_rgistration= np.random.exponential(0.5,5000)

sns.distplot(service_rgistration)

rarrival_reg= FloatVector(arrival_registration)
rservice_reg= FloatVector(service_rgistration)

simulation_reg= r['queue_step'](rarrival_reg,rservice_reg,2)
simulation_reg[2]['waiting'].mean()

simulation_reg[2]

randomization= np.random.random(5000)

departures= simulation_reg[2]['departures']

teller_arrival=departures[randomization<= 0.65]
len(teller_arrival)
cs_arrival= departures[randomization > 0.65]
len(cs_arrival)
teller_service= np.random.normal(10,2,3242)
cs_service= np.random.exponential(13,1758)

rteller_arival= FloatVector(teller_arrival)
rcs_arival= FloatVector(cs_arrival)
rteller_service=FloatVector(teller_service)
rcs_service=FloatVector(cs_service)

sim_teller=r['queue_step'](rteller_arival,rteller_service,20)

sim_cs=r['queue_step'](rcs_arival,rcs_service,15)

waiting_reg=simulation_reg[2]['waiting'].mean()
waiting_teller=sim_teller[2]['waiting'].mean()
waiting_cs=sim_cs[2]['waiting'].mean()

waiting_reg+((waiting_teller+waiting_cs)/2)
















































































    
    
    
    
    






































    
    
    
    

























        
        
    
    
    






