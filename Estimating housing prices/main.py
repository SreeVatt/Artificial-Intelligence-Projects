# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:04:56 2024

@author: sreev
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

housing_data=datasets.fetch_california_housing()

x,y=shuffle(housing_data.data,housing_data.target,random_state=7)
num_training=int(0.8*len(x))
x_train,y_train=x[:num_training],y[:num_training]
x_test,y_test=x[num_training:],y[num_training:]

#Perfroming Decision Tree Regressor

dt_regressor=DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(x_train,y_train)

#Performing AdaBoostRegeressor

ab_regressor=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
ab_regressor.fit(x_train,y_train)

#Displaying Result of Decision Tree
y_pred_dt=dt_regressor.predict(x_test)
mse=mean_squared_error(y_test, y_pred_dt)
evs=explained_variance_score(y_test, y_pred_dt)
print("\n\t\tDecision Tree\n")
print("Mean Squared Error : ",mse)
print("Explained Variable Score : ",evs)

#Displaying Result of AdaBoost
y_pred_ab=ab_regressor.predict(x_test)
mse=mean_squared_error(y_test, y_pred_ab)
evs=explained_variance_score(y_test, y_pred_ab)
print("\n\t\tAdaBoost\n")
print("Mean Squared Error : ",mse)
print("Explained Variable Score : ",evs)

 
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

plot_feature_importance(dt_regressor.feature_importances_,'Decision Tree regressor', housing_data.feature_names)
plot_feature_importance(ab_regressor.feature_importances_,'AdaBoost regressor', housing_data.feature_names)
    
