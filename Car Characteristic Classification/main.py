# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:24:09 2024

@author: sreev
"""

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


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
y = label_encoder[-1].fit_transform(x[:, -1]).astype(int)

#Building Random Forest Classifier
params={'n_estimators':200,'max_depth':8,'random_state':7}
classifier=RandomForestClassifier(**params)
classifier.fit(x,y)

#CrossValidation
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(classifier,x,y,scoring='accuracy',cv=3)
print("Accuracy of the Classifier :",round(100*accuracy.mean(),2),'%')          

#Testing
input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low']
input_data_encoded = np.empty(len(input_data))  # Create an empty array to store encoded values

for i, item in enumerate(input_data):
    input_data_encoded[i] = label_encoder[i].transform([item])[0]  # Encode and store in the array

input_data_encoded = np.array(input_data_encoded)
input_data_encoded = input_data_encoded.reshape(1, -1)


# Predict and print output for a particular datapoint
output_class = classifier.predict(input_data_encoded)
print ("Output class:",label_encoder[-1].inverse_transform(output_class)[0])

#Validation Curves
from sklearn.model_selection import validation_curve
classifier=RandomForestClassifier(max_depth=4,random_state=7)
parameter_grid=np.linspace(25,200,8).astype(int)
train_scores,validation_scores=validation_curve(classifier,x,y,"n_estimators",parameter_grid,cv=5)
print ("\n##### VALIDATION CURVES #####")
print ("\nParam: n_estimators\nTraining scores:\n", train_scores)
print ("\nParam: n_estimators\nValidation scores:\n",validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1),
color='black')
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

#Validation Curves
classifier = RandomForestClassifier(n_estimators=20,random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)
train_scores, valid_scores = validation_curve(classifier, x, y,"max_depth", parameter_grid, cv=5)
print ("\nParam: max_depth\nTraining scores:\n", train_scores)
print ("\nParam: max_depth\nValidation scores:\n",validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1),
color='black')
plt.title('Validation curve')
plt.xlabel('Maximum depth of the tree')
plt.ylabel('Accuracy')
plt.show()

 # Learning curves
from sklearn.learning_curve import learning_curve
classifier = RandomForestClassifier(random_state=7)
parameter_grid = np.array([200, 500, 800, 1100])
train_sizes, train_scores, validation_scores =learning_curve(classifier,x, y, train_sizes=parameter_grid, cv=5)
print ("\n##### LEARNING CURVES #####")
print ("\nTraining scores:\n", train_scores)
print ("\nValidation scores:\n", validation_scores)

