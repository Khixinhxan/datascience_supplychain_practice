#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:23:49 2020

@author: haythamomar
"""

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


rfm= pd.read_csv('rfm.csv')
rfm.columns

X= rfm[['frequency','monetary','recency']]

km= KMeans(n_clusters=3,n_init= 10,max_iter=300,tol=0.0001)

fitting= km.fit_predict(X)

X['centroids']=fitting

sns.pairplot(data=X,hue='centroids')


sse= []

for k in range(1,11):
    kmeans= KMeans(n_clusters=k,n_init= 10,max_iter=300,tol=0.0001)
    a= kmeans.fit(X)
    sse.append(a.inertia_)
    
    
sse

plt.plot(range(1,11),sse)


#### regression

retail_clean= pd.read_csv('retail_clean.csv')
retail_clean.columns
retail_clean['InvoiceDate']= pd.to_datetime(retail_clean['InvoiceDate'])
retail_clean['date']= retail_clean['InvoiceDate'].dt.strftime('%Y-%m-%d')

retail_clean['date']= pd.to_datetime(retail_clean['date'])

daily_revenue=retail_clean.groupby(['date']).agg(total_revenue= ('Revenue','sum')).reset_index()

daily_revenue['month']= daily_revenue['date'].dt.month
daily_revenue['dayofweek']=daily_revenue['date'].dt.dayofweek
daily_revenue['trend']= range(1, daily_revenue.shape[0]+1)

daily_revenue['month']= daily_revenue['month'].astype('category')

weekdays= {0: 'Monday',1:'Tuesday',2:'Wednesday',3: 'Thursday',4: 'Friday',
           5: 'Saturday',6: 'Sunday'}

daily_revenue['dayofweek1']=daily_revenue['dayofweek'].map(weekdays)


daily_revenue= daily_revenue.drop('dayofweek',axis=1)


daily_Revenue_encoded= pd.get_dummies(daily_revenue)


daily_Revenue_encoded.columns

plt.plot(daily_revenue['date'],daily_revenue['total_revenue'])


from sklearn.model_selection import train_test_split

daily_Revenue_encoded.shape

X= daily_Revenue_encoded.drop(['date','total_revenue'],axis=1).values
y= daily_Revenue_encoded['total_revenue'].values

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,shuffle=False)

len(X_train)
len(X_test)


from sklearn.linear_model import LinearRegression , Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor


model_linear= LinearRegression().fit(X_train,y_train)
model_lasso= Lasso(alpha=0.006,normalize=True,tol=0.000001,max_iter=1000).fit(X_train,y_train)
model_tree= DecisionTreeRegressor().fit(X_train,y_train)
model_knn= KNeighborsRegressor(n_neighbors=3).fit(X_train,y_train)
###Training Score
model_linear.score(X_train,y_train)
model_lasso.score(X_train,y_train)
model_tree.score(X_train,y_train)
model_knn.score(X_train,y_train)
### Testing Score

model_linear.score(X_test,y_test)
model_lasso.score(X_test,y_test)
model_tree.score(X_test,y_test)
model_knn.score(X_test,y_test)

##Predictions
y_linear= model_linear.predict(X_test)
y_lasso= model_lasso.predict(X_test)
y_tree= model_tree.predict(X_test)
y_knn= model_knn.predict(X_test)
###mean squared error
mean_squared_error(y_test,y_linear)
mean_squared_error(y_test,y_lasso)
mean_squared_error(y_test,y_tree)
mean_squared_error(y_test,y_knn)
### mean absolute error

mean_absolute_error(y_test,y_linear)
mean_absolute_error(y_test,y_lasso)
mean_absolute_error(y_test,y_tree)
mean_absolute_error(y_test,y_knn)


#### parameter tuning 

MAE_training=[]
MAE_testing=[]
neighbors=range(1,20)

for n in neighbors:
    model= KNeighborsRegressor(n_neighbors=n).fit(X_train,y_train)
    y_predict_training= model.predict(X_train)
    y_predict_testing=model.predict(X_test)
    training= mean_absolute_error(y_predict_training,y_train)
    testing=mean_absolute_error(y_predict_testing,y_test)
    MAE_training.append(training)
    MAE_testing.append(testing)
    
    
plt.plot(neighbors,MAE_training,label='training')
plt.plot(neighbors,MAE_testing,label='testing')
plt.legend(loc='upperleft')


import numpy as np
    
 
MAE_training=[]
MAE_testing=[]  
model_scores=[]  
 
alphas= np.linspace(0, 1,100)
   
for alpha in alphas:
    model= Lasso(alpha=alpha,normalize=True,fit_intercept=True,
                 max_iter=20000).fit(X_train,y_train)
    y_predict_training= model.predict(X_train)
    y_predict_testing=model.predict(X_test)
    scores= model.score(X_train,y_train)
    training= mean_absolute_error(y_predict_training,y_train)
    testing=mean_absolute_error(y_predict_testing,y_test)
    MAE_training.append(training)
    MAE_testing.append(testing)
    model_scores.append(scores)



plt.plot(alphas,MAE_training,label='training')
plt.plot(alphas,MAE_testing,label='testing')
plt.legend(loc='upperleft')

alpha_data= pd.DataFrame({'alpha':alphas,'training': MAE_training,
                          'testing':  MAE_testing,'scores':model_scores})


alpha_data[alpha_data.scores==max(alpha_data.scores)]

model_alpha= Lasso(alpha=20).fit(X_train,y_train)

model_alpha.coef_

names= daily_Revenue_encoded.drop(['date','total_revenue'],axis=1).columns

plt.plot(names,model_alpha.coef_)
plt.xticks(rotation=90)

data_ceof= pd.DataFrame({'names':names,'coef': model_alpha.coef_})



### classfication 

# Attribute Information:

# Input variables:
# # bank client data:
# 1 - age (numeric)
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# # related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular','telephone') 
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# # other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# # social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 20 - nr.employed: number of employees - quarterly indicator (numeric)

# Output variable (desired target):
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')





import pandas as pd
import seaborn as sns
import numpy as np
banking= pd.read_csv('bank-full.csv')

banking.iloc[0,:]


banking.y.value_counts()


banking.info()


sns.pairplot(banking.iloc[:,[0,5,11,12,13,14,16]],hue='y')



data=banking.iloc[:,[0,5,11,12,13,14]].corr()

sns.heatmap(data)



### mapping



banking= pd.read_csv('bank-full.csv')

dict_target= {'yes':1,'no':0}

banking['target']= banking['y'].map(dict_target)
banking= banking.drop('y',axis=1)

y= banking['target'].values

X_= banking.drop('target',axis=1)
X= pd.get_dummies(X_).values
X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=12)

## without paremeter tune 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=12)

model_lr= LogisticRegression()

model_lr.fit(X_train,y_train)


##training score
model_lr.score(X_train,y_train)

## testing score

model_lr.score(X_test,y_test)




####Pre Processing

banking.describe()

from sklearn.preprocessing import MinMaxScaler


scaler= MinMaxScaler(feature_range=(0,1))

scaler.fit(X_train)

X_train=scaler.transform(X_train)
X_test= scaler.transform(X_test)



model_lr= LogisticRegression()

model_lr.fit(X_train,y_train)


##training score
model_lr.score(X_train,y_train)

## testing score

model_lr.score(X_test,y_test)


### Grid Search CV

from sklearn.model_selection import GridSearchCV


Cs= np.logspace(-8, 5,20)

grid= {'C': Cs,'penalty': ['l1','l2']}


model_grid= LogisticRegression()

grid_fit= GridSearchCV(estimator=model_grid, param_grid=grid, cv=6,scoring='roc_auc')

grid_fit.fit(X_train,y_train)

grid_fit.best_params_
grid_fit.best_score_


### Area Under the Curve
import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
## pip install scikit-plot


model= LogisticRegression(C=20691.3808111479,penalty='l2')
model.fit(X_train,y_train)
y_predict= model.predict(X_test)
y_predict
print(confusion_matrix( y_test, y_predict))

y_test

## frequency

np.unique(y_test,return_counts=True)

tn,fp,fn,tp= confusion_matrix(y_test,y_predict).ravel()

(tn,fp,fn,tp)

y_predicted_probability= model.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_predicted_probability)

y_predicted_probability_both=model.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test,y_predicted_probability_both)




### pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
### prepration for pipelines
banking= pd.read_csv('bank-full.csv')

dict_target= {'yes':1,'no':0}

banking['target']= banking['y'].map(dict_target)
banking= banking.drop('y',axis=1)

y= banking['target'].values

X_= banking.drop('target',axis=1)
X= pd.get_dummies(X_).values
X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=12)





p_lr= Pipeline([('Imputing',SimpleImputer(missing_values= np.nan,strategy= 'mean')),
                ('scaling',StandardScaler()),
                ('logistic',LogisticRegression())])


p_rf= Pipeline([('Imputing',SimpleImputer(missing_values= np.nan,strategy= 'mean')),
                ('scaling',StandardScaler()),
                ('rf' ,RandomForestClassifier())])
p_svc= Pipeline([('Imputing',SimpleImputer(missing_values= np.nan,strategy= 'mean')),
                ('scaling',StandardScaler()),
                ('SVC',SVC())])
p_KNN= Pipeline([('Imputing',SimpleImputer(missing_values= np.nan,strategy= 'mean')),
                ('scaling',StandardScaler()),
                ('knn',KNeighborsClassifier())])

param_range= [1,2,3,4,5,6,7,8,9,10]
lr_range= np.logspace(-5,5,15)


grid_logistic= [{'logistic__penalty':['l1','l2'],
                 'logistic__C': lr_range,
                 'logistic__solver': ['liblinear']}]

grid_rf= [{'rf__criterion': ['gini','entropy'],
           'rf__min_samples_leaf': param_range}]

grid_svc= [{'SVC__kernel': ['linear','rbf'],
            'SVC__C': param_range}]

grid_knn= [{'knn__n_neighbors': param_range}]

pipes= [p_lr,p_KNN]
grids=[grid_logistic,grid_knn]

fitted_parms=[]
fitted_score=[]
fitted_roc=[]
n_jobs=-1
for i in range(0,2):
    model= GridSearchCV(pipes[i], grids[i], cv=3,scoring='accuracy',verbose=10)
    model.fit(X_train,y_train)
    y_pred_prob= model.predict_proba(X_test)[:,1]
    roc= roc_auc_score( y_test, y_pred_prob)
    fitted_parms.append(model.best_params_)
    fitted_score.append(model.best_score_)
    fitted_roc.append(roc)

    
fitted_parms
fitted_score 
fitted_roc   

from random import randint



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

param_dist = {"max_depth": [3, None],
          
           "min_samples_leaf": range(1,9),
             "criterion": ["gini", "entropy"]}

tree= DecisionTreeClassifier()
rf= RandomForestClassifier()

tree.fit(X_train, y_train)
tree.score(X_train,y_train)
predict_tree=tree.predict_proba(X_test)[:,1]
roc_auc_score(y_test,predict_tree)

rf.fit(X_train,y_train)
rf.score(X_train,y_train)
predict_rf=rf.predict_proba(X_test)[:,1]
roc_auc_score(y_test,predict_rf)

tree_cv= RandomizedSearchCV(tree, param_dist,cv=5)
rf_cv=RandomizedSearchCV(rf, param_dist,cv=5)

tree_cv.fit(X_train,y_train)
rf_cv.fit(X_train,y_train)

tree_cv.best_score_
rf_cv.best_score_

cv_tree_predict_prob= tree_cv.predict_proba(X_test)[:,1]
roc_auc_score(y_test,cv_tree_predict_prob )

cv_rf_predict_prob=rf_cv.predict_proba(X_test)[:,1]
roc_auc_score(y_test,cv_rf_predict_prob )























    




























   
    
    
    
    
    
    











































































