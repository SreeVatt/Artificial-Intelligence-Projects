# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:29:07 2024

@author: sreev
"""
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import numpy as np

input_file='.\\Data\\adult.data'
#Read The Data
x=[]
y=[]
count_lessthan50k=0
count_morethan50k=0
num_images_threshold=1000

with open(input_file,'r') as f:
    for line in f.reaflines():
        if '?' in line:
            continue
        data=line[:1].split(',')
        if data[-1]=='<=50K' and count_lessthan50k<num_images_threshold:
            x.append(data)
            count_lessthan50k=count_lessthan50k+1
        elif data[-1]=='>50K' and count_morethan50k<num_images_threshold:
            x.append(data)
        if count_lessthan50k>=num_images_threshold and count_morethan50k>= num_images_threshold:
            break
x=np.array(x)

#Convert string data into numerical data
label_encoder=[]
x_encoded=np.empty(x.shape)
for i,item in enumerate(x[0]):
    if item.isdigit():
        x_encoded[:,i]=x[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        x_encoded[:,i]=label_encoder[-1].fit_transform(x[:,i])
x=x_encoded[:,:-1].astype(int)
y=x_encoded[:,-1].astype(int)
    
#Building Classifier
classifier_gaussiannb=GaussianNB()
classifier_gaussiannb.fit(x,y)

#Cross Validation
from sklearn.model_selection import cross_validate,cross_val_score
X_train, X_test, y_train, y_test =cross_validate.train_test_split(x, y, test_size=0.25,random_state=5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)

# compute F1 score of the classifier
f1 = cross_val_score(classifier_gaussiannb,x, y, scoring='f1_weighted', cv=5)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")