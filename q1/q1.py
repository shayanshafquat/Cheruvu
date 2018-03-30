# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:00:42 2018

@author: SHAYAN
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def preprocessing(filepath):
    df = pd.read_csv(filepath)
    
    sav_trans = savgol_filter(df[df.columns[1:3579]],9,6)
    df.drop(df.columns[1:3579],axis=1,inplace=True)
        
    lb = LabelBinarizer()
    df['Depth'] = lb.fit_transform(df['Depth'])
    index = df['PIDN']
    
    if 'train' in filepath:
        y = df[['Ca','P','pH','SOC','Sand']].values
        df.drop(['Ca','P','pH','SOC','Sand'],axis=1,inplace=True)
    X = np.concatenate((df[df.columns[1:]].values,sav_trans),axis=1)
    
    if 'train' in filepath:
        return X,y,index
    else:
        return X,index

#clf = linear_model.Lasso(alpha=0.1)
reg = SVR(C=8000.0)
clf = RandomForestRegressor(n_estimators=50)

X_train,y_train,_ = preprocessing('train.csv')
X_test,index = preprocessing('test.csv')

'''
X_train,X_test,y_train,y_test= train_test_split(X_train,y_train,test_size=0.2,random_state=1)
clf.fit(X_train,y_train)
pre =clf.predict(X_test)
print(r2_score(y_test,pre))
'''

preds = np.zeros((X_test.shape[0], 5))
for i in range(5):
    clf.fit(X_train,y_train[:,i])
    reg.fit(X_train,y_train[:,i])
    preds[:,i] = (1/2)*(clf.predict(X_test)+reg.predict(X_test))
    
soln = pd.DataFrame(preds,index,columns=['Ca','P','pH','SOC','Sand'])

soln.to_csv('soln.csv',index_label=['PIDN'])
