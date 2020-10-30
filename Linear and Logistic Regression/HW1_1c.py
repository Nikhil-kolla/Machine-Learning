#!/usr/bin/env python
# coding: utf-8

# # BEST FIT LINE

# In[355]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('D:\\DataSets\\data.csv')
print(df.shape)


# In[356]:


input_features=df.drop(['Body_Weight'],axis=1)
y=df['Body_Weight']


# In[357]:


input_features = input_features.as_matrix()
y = y.as_matrix()


# In[358]:


def rmse(x,y,w):
    temp = x.dot(w)
    temp = np.subtract(y,temp)
    temp = temp**2
    answer = np.sum(temp)
    answer = np.divide(answer,len(x))
    answer = np.sqrt(answer)
    return answer


# In[359]:


def cost(x,y,w):
    total_loss = 0
    for i in range(0,len(x)):
        loss = y[i]-x[i].dot(w)
        loss = 2 * loss
        loss = loss * x[i]
        total_loss+=loss
    return total_loss


# In[360]:


def gradient_descent(x,y):
    learning_rate = 0.00002
    error_t = []
    error_v = []
    w_old = np.zeros(1)
    old_cost = rmse(x,y,w)
    #print(old_cost)
    for i in range(100):
        w_new = w_old + (1/(2*len(x))*(learning_rate * (cost(x,y,w_old))))
        error_t.append(rmse(x,y,w_new))
        new_cost = rmse(x,y,w_new)
        result = np.subtract(old_cost,new_cost)
        if (result < 0.1):
            return w_old,error_t
            break;
        w_old = w_new 
    print('Optimal Iteration ',i+1)
    print('Optimal w ',w_old)
    return w_old,error_t


# In[365]:


w,error = gradient_descent(input_features,y)


# In[366]:


print(input_features.shape)
print(w.shape)


# In[367]:


y_obtained_1 = input_features.dot(w)
#print(y_obtained)


# In[368]:


from matplotlib import pyplot as plt
plt.scatter(input_features,y,label='give data')
plt.plot(input_features,y_obtained_1,color='red',label='Best fit')
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.title('Without Regularization')
plt.legend(loc='upper left')
plt.show()


# # With L1 Regularization

# In[369]:


def rmse(x,y,w):
    temp = x.dot(w)
    temp = np.subtract(y,temp)
    temp = temp**2
    answer = np.sum(temp)
    answer = np.divide(answer,len(x))
    answer = np.sqrt(answer)
    return answer


# In[370]:


def cost(x,y,w):
    total_loss = 0
    for i in range(0,len(x)):
        loss = y[i]-x[i].dot(w)
        loss = 2 * loss
        loss = loss * x[i]
        total_loss+=loss
    temp = (10000)*w
    total_loss = total_loss+temp    
    return total_loss


# In[376]:


def gradient_descent(x,y):
    learning_rate = 0.00002
    error_t = []
    error_v = []
    w_old = np.zeros(1)
    old_cost = rmse(x,y,w)
    #print(old_cost)
    for i in range(100):
        w_new = w_old + (1/(2*len(x))*(learning_rate * (cost(x,y,w_old))))
        error_t.append(rmse(x,y,w_new))
        new_cost = rmse(x,y,w_new)
        result = np.subtract(old_cost,new_cost)
        if (result < 0.1):
            return w_old,error_t
            break;
        w_old = w_new 
    print('Optimal Iteration ',i+1)
    print('Optimal w ',w_old)
    return w_old,error_t


# In[377]:


w,error = gradient_descent(input_features,y)


# In[379]:


y_obtained_2 = input_features.dot(w)


# In[380]:


from matplotlib import pyplot as plt
plt.scatter(input_features,y,label='give data')
plt.plot(input_features,y_obtained_2,color='yellow',label='L1 reg')
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.title('L1 Regularization')
plt.legend(loc='upper left')
plt.show()


# # With L2 Regularization

# In[381]:


def rmse(x,y,w):
    temp = x.dot(w)
    temp = np.subtract(y,temp)
    temp = temp**2
    answer = np.sum(temp)
    answer = np.divide(answer,len(x))
    answer = np.sqrt(answer)
    return answer


# In[392]:


def cost(x,y,w):
    total_loss = 0
    for i in range(0,len(x)):
        loss = y[i]-x[i].dot(w)
        loss = 2 * loss
        loss = loss * x[i]
        total_loss+=loss
    temp = (10)*(w**2)
    total_loss = total_loss+temp    
    return total_loss


# In[393]:


def gradient_descent(x,y):
    learning_rate = 0.00002
    error_t = []
    error_v = []
    w_old = np.zeros(1)
    old_cost = rmse(x,y,w)
    #print(old_cost)
    for i in range(100):
        w_new = w_old + (1/(2*len(x))*(learning_rate * (cost(x,y,w_old))))
        error_t.append(rmse(x,y,w_new))
        new_cost = rmse(x,y,w_new)
        result = np.subtract(old_cost,new_cost)
        if (result < 0.1):
            return w_old,error_t
            break;
        w_old = w_new 
    print('Optimal Iteration ',i+1)
    print('Optimal w ',w_old)
    return w_old,error_t


# In[395]:


w,error = gradient_descent(input_features,y)


# In[396]:


y_obtained_3 = input_features.dot(w)


# In[397]:


from matplotlib import pyplot as plt
plt.scatter(input_features,y,label='give data')
plt.plot(input_features,y_obtained_3,color='green',label='L2 reg')
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.title('L2 Regularization')
plt.legend(loc='upper left')
plt.show()


# # Comparision

# In[398]:


from matplotlib import pyplot as plt
plt.scatter(input_features,y,label='give data',color='yellow')
plt.plot(input_features,y_obtained_1,color='red',label='Best fit')
plt.plot(input_features,y_obtained_2,color='blue',label='L1 reg')
plt.plot(input_features,y_obtained_3,color='green',label='L2 reg')
plt.xlabel('Brain Weight')
plt.ylabel('Body Weight')
plt.title('Comparision')
plt.legend(loc='upper left')
plt.show()


# In[ ]:




