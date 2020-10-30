#!/usr/bin/env python
# coding: utf-8

# In[387]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[388]:


data_train = pd.read_csv('D:\\DataSets\\Logistic Regression\\train.csv')
print(data_train.shape)

ones = np.ones([len(data_train),1])
data_train_list = [pd.DataFrame(ones),data_train]
data_train = pd.concat(data_train_list,axis=1)

print(data_train.shape)


# In[389]:


data_test = pd.read_csv('D:\\DataSets\\Logistic Regression\\test.csv')
print(data_test.shape)

ones = np.ones([len(data_test),1])
data_test_list = [pd.DataFrame(ones),data_test]
data_test = pd.concat(data_test_list,axis=1)
print(data_test.shape)


# In[390]:


data_train.columns=['ones','age','workclass','fnlwgt','education','education-num','marital-status'
                   ,'occupation','relationship','race','sex','capital-gain','capital-loss'
                   ,'hours-per-week','native-country','>50K, <=50K']

data_test.columns=['ones','age','workclass','fnlwgt','education','education-num','marital-status'
                   ,'occupation','relationship','race','sex','capital-gain','capital-loss'
                   ,'hours-per-week','native-country','>50K, <=50K']
print(data_train.shape)
print(data_test.shape)


# In[391]:


categorical_features=['workclass','education','marital-status','occupation','relationship'
                     ,'race','sex','native-country','>50K, <=50K']


# In[392]:


from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()

for c in categorical_features:
    data_train[c] = encode.fit_transform(data_train[c])


# In[393]:


data_train.head()


# In[394]:


for c in categorical_features:
    data_test[c] = encode.fit_transform(data_test[c])


# In[395]:


data_test.head()


# In[396]:


x_train = data_train.drop(['>50K, <=50K'],axis=1)
y_train = data_train['>50K, <=50K']
print(x_train.shape)
print(y_train.shape)


# In[397]:


y_train= pd.factorize(y_train)[0]
y_train.shape


# In[398]:


y_test= pd.factorize(y_test)[0]
y_test.shape


# In[399]:


x_test = data_test.drop(['>50K, <=50K'],axis=1)
y_test = data_test['>50K, <=50K']
print(x_test.shape)
print(y_test.shape)


# In[400]:


x_train.shape


# In[401]:


from sklearn import preprocessing
mmscale = preprocessing.MinMaxScaler(feature_range=(0,1))

x_train = mmscale.fit_transform(x_train)
#x_train = x_train.as_matrix()
x_train


# In[402]:


x_test.shape


# In[403]:


from sklearn import preprocessing
mmscale1 = preprocessing.MinMaxScaler(feature_range=(0,1))

x_test = mmscale1.fit_transform(x_test)
#x_train = x_train.as_matrix()
x_test


# In[404]:


from random import random
w_old = np.random.random(15)
w_old


# In[405]:


x_train_1 = x_train[0:21112]
x_validation = x_train[21112:]
x_train = x_train_1
print(x_train.shape)
print(x_validation.shape)


# In[406]:


y_train_1 = y_train[:21112]
y_validation = y_train[21112:]
y_train = y_train_1
print(y_train.shape)
print(y_validation.shape)


# In[407]:


print(x_train.shape)
print(w_old.shape)


# In[408]:


def sigmoid(x):
    y_hat=np.zeros(len(x))
    for i in range(len(x)):
        y_hat[i] = 1 / (1 + np.exp(-x[i]))
    return y_hat    


# In[409]:


def hypothesis(x,w):
    z=np.dot(x,w)
    return sigmoid(z)


# In[410]:


def loss(predicted_y,y):
    return (-y*np.log(predicted_y)-(1-y)*np.log(1-predicted_y)).mean()


# In[411]:


def accuracy(x,y,w):
    a = hypothesis(x,w)
    for i in range(len(y)):
        if(a[i]>=0.5):
            a[i]=1
        if(a[i]<0.5):
            a[i]=0
    count = 0
    for i in range(len(a)):
        if(a[i]==y[i]):
            count+=1
    return (count*100)/len(y)


# In[412]:


def gradient_descent(x,y,w):
    learning_rate = 0.04
    w_old = w
    error_l=[]
    accuracy_l=[]
    for i in range(100):
        h = hypothesis(x,w_old)
        derivative = np.dot(x.T,(h-y))/y.size
        old_loss = loss(hypothesis(x,w_old),y)
#         print('Initial old loss ')
#         print(old_loss)
        w_new = w_old - (learning_rate*derivative)
        new_loss = loss(hypothesis(x,w_new),y)
#         print('Iteration ',i+1)
        error_l.append(np.sqrt(new_loss))
        accuracy_l.append(accuracy(x,y,w_new))
        
#         if(old_loss-new_loss<0.001):
#             print('Iteration ',i+1)
#             print('Optimal w ')
#             print(w_new)
#             return w_new,error_l,accuracy_l
#             break
        w_old = w_new
    return w_new,error_l,accuracy_l
        


# In[413]:


w_old = np.random.random(15)


# In[414]:


w_train_opt,train_error_list,train_accuracy_list = gradient_descent(x_train,y_train,w_old)
print(train_error_list)
print(len(train_error_list))
print(train_accuracy_list)
print(len(train_accuracy_list))


# In[415]:


w_validation_opt,validation_error_list,validation_accuracy_list = gradient_descent(x_validation,y_validation,w_train_opt)


# In[416]:


print(validation_error_list)
print(len(validation_error_list))
print(validation_accuracy_list)
print(len(validation_accuracy_list))


# In[417]:


w_test_opt,test_error_list,test_accuracy_list = gradient_descent(x_test,y_test,w_train_opt)


# In[418]:


print(validation_error_list)
print(len(validation_error_list))
print(validation_accuracy_list)
print(len(validation_accuracy_list))


# # L1 Regularization

# In[355]:


def sigmoid(x):
    y_hat=np.zeros(len(x))
    for i in range(len(x)):
        y_hat[i] = 1 / (1 + np.exp(-x[i]))
    return y_hat 


# In[356]:


def hypothesis(x,w):
    z=np.dot(x,w)
    return sigmoid(z)


# In[357]:


def loss(predicted_y,y,w):
    a = (-y*np.log(predicted_y)-(1-y)*np.log(1-predicted_y)).mean()
    temp = (1 *(np.sum(w)))/len(y)
    return a+temp


# In[362]:


def accuracy(x,y,w):
    a = hypothesis(x,w)
    for i in range(len(y)):
        if(a[i]>=0.5):
            a[i]=1
        if(a[i]<0.5):
            a[i]=0
    count = 0
    for i in range(len(a)):
        if(a[i]==y[i]):
            count+=1
    return (count*100)/len(y)


# In[363]:


def gradient_descent(x,y,w):
    learning_rate = 0.04
    w_old = w
    error_l=[]
    accuracy_l=[]
    for i in range(100):
        h = hypothesis(x,w_old)
        derivative = np.dot(x.T,(h-y))/y.size
        old_loss = loss(hypothesis(x,w_old),y,w_old)
#         print('Initial old loss ')
#         print(old_loss)
        a = (-1)*(w_old/len(x))
        w_new = w_old - (learning_rate*(derivative+a))
        new_loss = loss(hypothesis(x,w_new),y,w_new)
#         print('Iteration ',i+1)
        error_l.append(np.sqrt(new_loss))
        accuracy_l.append(accuracy(x,y,w_new))
        
#         if(old_loss-new_loss<0.001):
#             print('Iteration ',i+1)
#             print('Optimal w ')
#             print(w_new)
#             return w_new,error_l,accuracy_l
#             break
        w_old = w_new
    return w_new,error_l,accuracy_l


# In[364]:


w_old = np.random.random(15)


# In[365]:


w_train_opt_l1,train_error_list_l1,train_accuracy_list_l1 = gradient_descent(x_train,y_train,w_old)
print(train_error_list_l1)
print(len(train_error_list_l1))
print(train_accuracy_list_l1)
print(len(train_accuracy_list_l1))


# In[366]:


w_validation_opt_l1,validation_error_list_l1,validation_accuracy_list_l1 = gradient_descent(x_validation,y_validation,w_train_opt)
print(validation_error_list_l1)
print(len(validation_error_list_l1))
print(validation_accuracy_list_l1)
print(len(validation_accuracy_list_l1))


# In[420]:


w_test_opt_l1,test_error_list_l1,test_accuracy_list_l1 = gradient_descent(x_test,y_test,w_train_opt)
print(test_error_list_l1)
print(len(test_error_list_l1))
print(test_accuracy_list_l1)
print(len(test_accuracy_list_l1))


# # L2 Regularization

# In[424]:


def sigmoid(x):
    y_hat=np.zeros(len(x))
    for i in range(len(x)):
        y_hat[i] = 1 / (1 + np.exp(-x[i]))
    return y_hat 


# In[425]:


def hypothesis(x,w):
    z=np.dot(x,w)
    return sigmoid(z)


# In[426]:


def loss(predicted_y,y,w):
    a = (-y*np.log(predicted_y)-(1-y)*np.log(1-predicted_y)).mean()
    temp = (1000 *(np.sum(w**2)))/len(y)
    return a+temp


# In[427]:


def accuracy(x,y,w):
    a = hypothesis(x,w)
    for i in range(len(y)):
        if(a[i]>=0.5):
            a[i]=1
        if(a[i]<0.5):
            a[i]=0
    count = 0
    for i in range(len(a)):
        if(a[i]==y[i]):
            count+=1
    return (count*100)/len(y)


# In[428]:


def gradient_descent(x,y,w):
    learning_rate = 0.04
    w_old = w
    error_l=[]
    accuracy_l=[]
    for i in range(100):
        h = hypothesis(x,w_old)
        derivative = np.dot(x.T,(h-y))/y.size
        old_loss = loss(hypothesis(x,w_old),y,w_old)
#         print('Initial old loss ')
#         print(old_loss)
        a = (-1000)*(w_old/len(x))
        w_new = w_old - (learning_rate*(derivative+a))
        new_loss = loss(hypothesis(x,w_new),y,w_new)
#         print('Iteration ',i+1)
        error_l.append(np.sqrt(new_loss))
        accuracy_l.append(accuracy(x,y,w_new))
        
#         if(old_loss-new_loss<0.001):
#             print('Iteration ',i+1)
#             print('Optimal w ')
#             print(w_new)
#             return w_new,error_l,accuracy_l
#             break
        w_old = w_new
    return w_new,error_l,accuracy_l


# In[429]:


w_old = np.random.random(15)


# In[430]:


w_train_opt_l2,train_error_list_l2,train_accuracy_list_l2 = gradient_descent(x_train,y_train,w_old)
print(train_error_list_l2)
print(len(train_error_list_l2))
print(train_accuracy_list_l2)
print(len(train_accuracy_list_l2))


# In[431]:


w_validation_opt_l2,validation_error_list_l2,validation_accuracy_list_l2 = gradient_descent(x_validation,y_validation,w_train_opt_l2)
print(validation_error_list_l2)
print(len(validation_error_list_l2))
print(validation_accuracy_list_l2)
print(len(validation_accuracy_list_l2))


# In[433]:


w_test_opt_l2,test_error_list_l2,test_accuracy_list_l2 = gradient_descent(x_test,y_test,w_train_opt_l2)
print(test_error_list_l2)
print(len(test_error_list_l2))
print(test_accuracy_list_l2)
print(len(test_accuracy_list_l2))


# # Graphs

# In[435]:


from matplotlib import pyplot as plt
plt.plot(train_error_list,label= 'without Reg.')
plt.plot(train_error_list_l1,label = 'l1 Reg')
plt.plot(train_error_list_l2,label = 'l2 Reg')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error vs. Iteration')
plt.grid()
plt.legend(loc='upper right')
plt.show()


# In[437]:


from matplotlib import pyplot as plt
plt.plot(train_accuracy_list,label= 'without Reg.')
plt.plot(train_accuracy_list_l1,label = 'l1 Reg')
plt.plot(train_accuracy_list_l2,label = 'l2 Reg')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Iteration')
plt.grid()
plt.legend(loc='upper left')
plt.show()


# In[ ]:




