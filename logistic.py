#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[23]:


# formula of sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    
    #initializing learning rate,number of iterations)
    def __init__(self, L=0.01, iters=1000):
        self.L = L
        self.iters = iters
        self.weights = None
        self.bias = None
        
    
    def g_d(self,X,y):
        #getting number of samples and features
        n_samples,n_features = X.shape
        
        # initializing weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
         
        # updating weights and bias iters times
        for _ in range(self.iters):
                
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.L*dw
            self.bias = self.bias - self.L*db
    
    # calculating y_pred (hypothesis) function
    def predict(self, X ):
        y_linear = np.dot(X,self.weights)+self.bias
        y_pred=sigmoid(y_linear)
        class_pred=[0 if y<=0.5 else 1 for y in y_pred]
        return class_pred


# In[34]:


#using sklearn dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
# spliting to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

#creating logisticregression type of object
clf = LogisticRegression(L=0.01)
clf.g_d(X_train,y_train)
y_pred = clf.predict(X_test)

#calculating the accuracy of thw model
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)


# In[ ]:




