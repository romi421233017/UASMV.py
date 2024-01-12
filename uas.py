# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:55:05 2024

@author: lenovo yoga 13
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from skimage import exposure

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
data = mnist.data.astype("uint8").values  # Convert DataFrame to numpy array
labels = mnist.target.astype("uint8")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# HOG Feature Extraction
def hog_features(data):
    hog_features_list = []
    for img in data:
        img_reshaped = img.reshape(28, 28)
        
        # Compute HOG features
        features, hog_image = hog(img_reshaped, orientations=8, pixels_per_cell=(4, 4),
                                  cells_per_block=(1, 1), visualize=True)
        
        hog_features_list.append(features)
    
    return np.array(hog_features_list)

X_train_hog = hog_features(X_train)
X_test_hog = hog_features(X_test)

# Train an SVM classifier
clf = svm.SVC()
clf.fit(X_train_hog, y_train)

# Predictions
y_pred = clf.predict(X_test_hog)

# Evaluate performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)