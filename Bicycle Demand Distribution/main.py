# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:10:01 2024

@author: sreev
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

def plot_feature_importance(feature_importance,title,feature_name):
    #Normalize the importance value
    feature_importance=100.0*(feature_importance/max(feature_importance))
    
    #Sort the index values and flip them for descending order
    index_sorted=np.flipud(np.argsort(feature_importance))
    
    #Center the location
    pos=np.arange(index_sorted.shape[0] )+0.5
    
    #plot the bar graph
    plt.figure()
    plt.bar(pos,feature_importance[index_sorted],align='center')
    plt.xticks(pos, feature_name[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()
    
def load_dataset(filename):
    with open(filename, 'r') as file:
        filereader = csv.reader(file, delimiter=',')
        x, y = [], []
        for row in filereader:
            x.append(row[2:13])
            y.append(row[-1])
    
    # Extract feature names
    feature_names = np.array(x[0])
    
    # Remove the first row because they are feature names
    return np.array(x[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

#Reading the data and shuffling
x,y,feature_names=load_dataset('.\\Dataset\\day.csv')
x,y=shuffle(x,y,random_state=7)

#Training Testing in 9:1
num_training = int(0.9*len(x))
x_train,y_train,x_test,y_test=x[:num_training],y[:num_training],x[num_training:],y[num_training:]

#Training Regressor
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
rf_regressor.fit(x_train,y_train)

y_pred=rf_regressor.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
evs=explained_variance_score(y_test, y_pred)
print ("\n#### Random Forest regressor performance ####")
print ("Mean squared error =", round(mse, 2))
print ("Explained variance score =", round(evs, 2))
plot_feature_importance(rf_regressor.feature_importances_,'Random Forest regressor', feature_names)