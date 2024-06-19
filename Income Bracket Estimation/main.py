# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:29:07 2024

@author: sreev
"""
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score

input_file = '.\\Data\\adult.data'
x = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 1000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        data = line.strip().split(', ')  # Adjusted to handle spaces after commas if any
        if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
            x.append(data)
            count_lessthan50k += 1
        elif data[-1] == '>50K' and count_morethan50k < num_images_threshold:
            y.append(data)  # Append to y for '>50K' case
            count_morethan50k += 1
        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break

# Convert lists to NumPy arrays
x = np.array(x)
y = np.array(y)

# Encode categorical data
label_encoders = []
x_encoded = np.empty(x.shape)
for i, item in enumerate(x[0]):
    if item.isdigit():
        x_encoded[:, i] = x[:, i].astype(float)  # Convert to float if numeric
    else:
        label_encoders.append(preprocessing.LabelEncoder())
        x_encoded[:, i] = label_encoders[-1].fit_transform(x[:, i])

x = x_encoded[:, :-1].astype(int)
y = x_encoded[:, -1].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

# Build and train classifier
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)

# Evaluate classifier using cross-validation
f1_scores = cross_val_score(classifier_gaussiannb, x, y, scoring='f1_weighted', cv=5)
print("F1 score: {:.2f}%".format(100 * f1_scores.mean()))
