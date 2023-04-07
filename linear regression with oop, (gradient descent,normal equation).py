#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


# In[26]:


class LinearRegression:
    
    #initializing learning rate,number of iterations)
    def __init__(self, L=0.01, iters=1000,):
        self.L=L
        self.iters=iters
        self.weight=None
        self.bias=None

    def g_d(self, X, y):
        
        #getting number of samples and features
        n_samples, n_features = X.shape
        
        # initializing weights and bias to zero
        self.weights = np.zeros(n_features)   
        self.bias = 0
        
        # updating weights and bias iters times
        for _ in range(self.iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.L * dw
            self.bias = self.bias - self.L * db
    
    # calculating y_prediction(hypothesis) function
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    # another algorithm to mimize cost function
    def norm_eq(self,X,y):
        n_samples, n_features = X.shape
        X_new = np.array([np.ones(n_features), X]).T
        X_new_transpose = X_new.T  
        best_params = np.linalg.inv(X_new_transpose.dot(X_new)).dot(X_new_transpose).dot(y) 
      
        return best_params
    


# In[28]:


#creating data set and spliting it to train and test sets
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

#creating linearregression object with a name of 'reg'
reg = LinearRegression(L=0.01)
reg.g_d(X_train,y_train)
predictions = reg.predict(X_test)

def cost(y_test, predictions):
    return np.mean((y_test-predictions)**2)


cost = cost(y_test, predictions)
print(cost)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
# chgitem vonc plot anem normal eq_i stacac funkcian plt.plot(X, ??,color="red")
plt.show()

#hi
# In[ ]:




