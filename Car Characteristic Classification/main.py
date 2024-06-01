# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:24:09 2024

@author: sreev
"""

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np


input_file="C:\\Users\\sreev\\Downloads\\Car Characteristic Classification\\Data\\car.data"
#Reading the dataset
x=[]
count=0
with open(input_file,'r') as f:
    for line in f.readlines():
        data=line.strip().split(',')
        x.append(data)

x=np.array(x)

#Convertion of String in numerics
label_encoder=[]
x_encoder=np.empty(x.shape)
for i,item in enumerate(x[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    x_encoder[:,i]=label_encoder[-1].fit_transform(x[:,i])
x=x_encoder[:,:-1].astype(int) 
y=x_encoder[:,-1].astype(int)

#Building Random Forest Classifier
params={'n_estimators':200,'max_depth':8,'random_state':7}
classifier=RandomForestClassifier(**params)
classifier.fit(x,y)

#CrossValidation
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(classifier,x,y,scoring='accuracy',cv=3)
print("Accuracy of the Classifier :",round(100*accuracy.mean(),2),'%')          

#Testing
input_data=['vhigh','vhigh','2','2','small','low']
input_data_encoded=[-1]*len(input_data)
for i,item in enumerate(input_data):
    input_data_encoded[i]=int(label_encoder[i].transform(input_data[i]))
input_data_encoder=np.array(input_data_encoded)
