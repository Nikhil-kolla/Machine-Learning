# -*- coding: utf-8 -*-
"""A3_Q1(final)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/189CJJOH5KESfEXPWdWnkakqYFlCSPLwN

**A3-Q1(a)**
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm_notebook 
from sklearn.neural_network import MLPClassifier as mlp
import time
import pandas as pd
import seaborn as sn
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

filename1 = 'MNIST_Subset.h5'  #change path
f = h5py.File(filename1, 'r')
a_group_key = list(f.keys())[0]
labels = list(f.keys())[1]
X = np.array(f[a_group_key])
Y = np.array(f[labels])
X=X.reshape(14251,784)

from collections import Counter
Counter(Y)

dic={7:0,9:1}

Y_ = []
for i in range(14251):
    Y_.append(dic[Y[i]])

Y_hot=[]
for i in range(14251):
    temp=[0]*2
    temp[dic[Y[i]]]=1
    Y_hot.append(temp)
Y_hot=np.array(Y_hot)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y_hot,random_state=0)
print(X_train.shape, X_test.shape, Y_train.shape,Y_test.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,random_state=0)
print(X_train.shape, X_val.shape, Y_train.shape,Y_val.shape)

X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X,Y_,random_state=0)
print(X_train.shape, X_test.shape, Y_train.shape,Y_test.shape)
X_train_, X_val_, Y_train_, Y_val_ = train_test_split(X_train_,Y_train_,random_state=0)
print(X_train.shape, X_val.shape, Y_train.shape,Y_val.shape)

class network:
    def __init__(self, W1, W2, W3, W4):
        self.W1 = W1.copy()
        self.W2 = W2.copy()
        self.W3 = W3.copy()
        self.W4 = W4.copy()
        self.B1 = np.zeros((1,100))
        self.B2 = np.zeros((1,50))
        self.B3 = np.zeros((1,50))
        self.B4 = np.zeros((1,2))
  
    def sigmoid(self, X):
        return 1.0/(1.0 + np.exp(-X))
  
    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1,1)
  
    def forward_pass(self, X):
        self.A1 = np.matmul(X,self.W1) + self.B1  
        self.H1 = self.sigmoid(self.A1)  
        self.A2 = np.matmul(self.H1, self.W2) + self.B2  
        self.H2 = self.sigmoid(self.A2)
        self.A3 = np.matmul(self.H2, self.W3) + self.B3  
        self.H3 = self.sigmoid(self.A3)
        self.A4 = np.matmul(self.H3, self.W4) + self.B4
        self.H4 = self.softmax(self.A4)  
        return self.H4
    
    def grad_sigmoid(self, X):
        return X*(1-X) 
  
    def grad(self, X, Y):
        self.forward_pass(X)
        m = X.shape[0]
        self.dA4 = self.H4 - Y
        
        self.dW4 = np.matmul(self.H3.T,self.dA4)
        self.dB4 = np.sum(self.dA4, axis=0).reshape(1,-1)
        self.dH3 = np.matmul(self.dA4, self.W4.T)
        self.dA3 = np.multiply(self.dH3, self.grad_sigmoid(self.H3)) 
        
        self.dW3 = np.matmul(self.H2.T,self.dA3)
        self.dB3 = np.sum(self.dA3,axis=0).reshape(1,-1)
        self.dH2 = np.matmul(self.dA3, self.W3.T)
        self.dA2 = np.multiply(self.dH2, self.grad_sigmoid(self.H2)) 
        
        self.dW2 = np.matmul(self.H1.T,self.dA2)
        self.dB2 = np.sum(self.dA2,axis=0).reshape(1,-1)
        self.dH1 = np.matmul(self.dA2, self.W2.T)
        self.dA1 = np.multiply(self.dH1, self.grad_sigmoid(self.H1))

        self.dW1 = np.matmul(X.T, self.dA1)    
        self.dB1 = np.sum(self.dA1, axis=0).reshape(1, -1)  
      
    def fit(self, X, Y,X_val,Y_val,X_test,Y_test, epochs=1, learning_rate=1, display_loss=False):
        if display_loss:
            loss = {}
            acc=[]
            loss_val = []
            acc_val=[]
            loss_test=[]
            acc_test=[]
    
        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            self.grad(X, Y)
        
            m = X.shape[0]
            self.W4 -= learning_rate * (self.dW4/m)
            self.B4 -= learning_rate * (self.dB4/m)
            
            self.W3 -= learning_rate * (self.dW3/m)
            self.B3 -= learning_rate * (self.dB3/m)
            
            self.W2 -= learning_rate * (self.dW2/m)
            self.B2 -= learning_rate * (self.dB2/m)
            
            self.W1 -= learning_rate * (self.dW1/m)
            self.B1 -= learning_rate * (self.dB1/m)

            if display_loss:
              #train data
              Y_pred = self.predict(X)
              loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)

              temp1 = np.argmax(Y, axis=1)
              temp2=np.argmax(Y_pred, axis=1)
              acc.append(accuracy_score(temp1,temp2))

              #validation data
              Y_pred_val = self.predict(X_val)
              loss_val.append(log_loss(np.argmax(Y_val, axis=1), Y_pred_val))

              temp3 = np.argmax(Y_val, axis=1)
              temp4=np.argmax(Y_pred_val, axis=1)
              acc_val.append(accuracy_score(temp3,temp4))

              #test data
              Y_pred_test = self.predict(X_test)
              loss_test.append( log_loss(np.argmax(Y_test, axis=1), Y_pred_test) )

              temp5 = np.argmax(Y_test, axis=1)
              temp6 = np.argmax(Y_pred_test, axis=1)
              acc_test.append(accuracy_score(temp5,temp6))

        if display_loss:
          #For training data
          plt.plot(list(loss.values()))
          plt.xlabel('Epochs')
          plt.ylabel('Log Loss')
          plt.title("Loss Vs Itrs of Training Data")
          plt.show()

          plt.plot(acc)
          plt.xlabel('Epochs')
          plt.ylabel('Accuracy')
          plt.title("Acc Vs Itrs of Training Data")
          plt.show()

          #For validation data
          plt.plot(loss_val)
          plt.xlabel('Epochs')
          plt.ylabel('Log Loss')
          plt.title("Loss Vs Itrs of Validation Data")
          plt.show()

          plt.plot(acc_val)
          plt.xlabel('Epochs')
          plt.ylabel('Accuracy')
          plt.title("Acc Vs Itrs of Validation Data")
          plt.show()

          #For Test data
          plt.plot(loss_test)
          plt.xlabel('Epochs')
          plt.ylabel('Log Loss')
          plt.title("Loss Vs Itrs of Test Data")
          plt.show()

          plt.plot(acc_test)
          plt.xlabel('Epochs')
          plt.ylabel('Accuracy')
          plt.title("Acc Vs Itrs of Test Data")
          plt.show()
        return self.W3
            
            
    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()

"""**A3_Q1(b)**"""

W1 = np.random.randn(784,100)
W2 = np.random.randn(100,50)
W3 = np.random.randn(50,50)
W4 = np.random.randn(50,2)

inn = time.time()
model = network(W1, W2,W3,W4)
weights = model.fit(X_train,Y_train,X_val,Y_val,X_test,Y_test,epochs=100,learning_rate=0.5,display_loss=True)
out = time.time()
print("Time taken by the model to train the data: ",(out-inn))

Y_pred_train = model.predict(X_train)
Y_pred_train = np.argmax(Y_pred_train,1)

Y_pred_val = model.predict(X_val)
Y_pred_val = np.argmax(Y_pred_val,1)

Y_pred_test = model.predict(X_test)
Y_pred_test = np.argmax(Y_pred_test,1)

accuracy_train = accuracy_score(Y_pred_train, Y_train_)
accuracy_val = accuracy_score(Y_pred_val, Y_val_)
accuracy_test = accuracy_score(Y_pred_test, Y_test_)

print("Training accuracy ",accuracy_train*100,"%")
print("Validation accuracy ",accuracy_val*100,"%")
print("Test accuracy ",accuracy_test*100,"%")

"""**A3_Q1(c)**"""

weights.shape

model_tSNE = TSNE(n_components=2, random_state=0)
tsne_data = model_tSNE.fit_transform(weights)

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2"))
sn.FacetGrid(tsne_df, size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

"""**A3_Q1(d)**"""

clf_with_three_hidden_layer = mlp(hidden_layer_sizes=(100,50,50), max_iter=100, verbose=1, solver='lbfgs',
                    learning_rate_init=0.5,activation='logistic',random_state=42)

clf_with_three_hidden_layer.fit(X_train,Y_train)

print("Training set score: ",clf_with_three_hidden_layer.score(X_train, Y_train)*100,"%")
print("Validation score: ",clf_with_three_hidden_layer.score(X_val,Y_val)*100,"%")
print("Test set score: ",clf_with_three_hidden_layer.score(X_test, Y_test)*100,"%")



