#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and loading Data

# In[539]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('D:\\DataSets\\abalone\\abalone\\Dataset.Data',delimiter=" ")


# In[540]:


print(df.shape)


# In[541]:


print(df.columns)


# # changing categorical values into numerical values

# In[542]:


df['sex'] = pd.factorize(df['sex'])[0]


# In[567]:


input_features=df.drop(['rings'],axis=1)
print(input_features.shape)
ones = np.ones([len(df),1])
print(ones.shape)
input_features_list = [pd.DataFrame(ones),input_features]
input_features = pd.concat(input_features_list,axis=1)
print(input_features.shape)
y=df['rings']
print(y.shape)


# In[568]:


input_features_set_1=input_features[0:835]
input_features_set_2=input_features[835:1670]
input_features_set_3=input_features[1670:2505]
input_features_set_4=input_features[2505:3340]
input_features_set_5=input_features[3340:]
y_set_1=y[:835]
y_set_2=y[835:1670]
y_set_3=y[1670:2505]
y_set_4=y[2505:3340]
y_set_5=y[3340:]


# In[569]:


input_features_set_1 = input_features_set_1.as_matrix()
input_features_set_2=input_features_set_2.as_matrix()
input_features_set_3=input_features_set_3.as_matrix()
input_features_set_4=input_features_set_4.as_matrix()
input_features_set_5=input_features_set_5.as_matrix()
y_set_1=y_set_1.as_matrix()
y_set_2=y_set_2.as_matrix()
y_set_3=y_set_3.as_matrix()
y_set_4=y_set_4.as_matrix()
y_set_5=y_set_5.as_matrix()


# # Root Mean Squared Error

# In[570]:


def rmse(x,y,w):
    temp = x.dot(w)
    temp = np.subtract(y,temp)
    temp = temp**2
    answer = np.sum(temp)
    answer = np.divide(answer,len(x))
    answer = np.sqrt(answer)
    return answer


# In[571]:


def cost(x,y,w):
    total_loss = 0
    for i in range(0,len(x)):
        loss = y[i]-x[i].dot(w)
        loss = -2 * loss
        loss = loss * x[i]
        total_loss += loss
    return total_loss


# In[572]:


def gradient_descent(x,y,x1,y1):
    learning_rate = 0.02
    error_t = []
    error_v = []
    w_old = np.zeros(9)
    for i in range(1000):
        w_new = w_old - (1/(2*len(x))*(learning_rate * (cost(x,y,w_old))))
        error_t.append(rmse(x,y,w_new))
        error_v.append(rmse(x1,y1,w_new))
        result = np.subtract(w_new,w_old)
        result = np.sum(result)
        if (result < 0.1):
            print('Optimal Iteration ',i+1)
            print('Optimal w ',w_old)
            return w_old,error_t,error_v
            break;
        w_old = w_new     


# In[573]:


#Keeping set_1 as validation set
#Concatenating all the remaining four sets of input features and y
x1 = np.concatenate((input_features_set_2,input_features_set_3),axis = 0)
x1 = np.concatenate((x1,input_features_set_4),axis = 0)
x1 = np.concatenate((x1,input_features_set_5),axis = 0)
y1 = np.concatenate((y_set_2,y_set_3),axis = 0)
y1 = np.concatenate((y1,y_set_4),axis=0)
y1 = np.concatenate((y1,y_set_5),axis=0)


# In[574]:


#Keeping set_2 as validation set
#Concatenating all the remaining four sets of input features and y
x2 = np.concatenate((input_features_set_1,input_features_set_3),axis = 0)
x2 = np.concatenate((x2,input_features_set_4),axis = 0)
x2 = np.concatenate((x2,input_features_set_5),axis = 0)
y2 = np.concatenate((y_set_1,y_set_3),axis = 0)
y2 = np.concatenate((y2,y_set_4),axis=0)
y2 = np.concatenate((y2,y_set_5),axis=0)


# In[575]:


#Keeping set_3 as validation set
#Concatenating all the remaining four sets of input features and y
x3 = np.concatenate((input_features_set_1,input_features_set_2),axis = 0)
x3 = np.concatenate((x3,input_features_set_4),axis = 0)
x3 = np.concatenate((x3,input_features_set_5),axis = 0)
y3 = np.concatenate((y_set_1,y_set_2),axis = 0)
y3 = np.concatenate((y3,y_set_4),axis=0)
y3 = np.concatenate((y3,y_set_5),axis=0)


# In[576]:


#Keeping set_4 as validation set
#Concatenating all the remaining four sets of input features and y
x4 = np.concatenate((input_features_set_1,input_features_set_2),axis = 0)
x4 = np.concatenate((x4,input_features_set_3),axis = 0)
x4 = np.concatenate((x4,input_features_set_5),axis = 0)
y4 = np.concatenate((y_set_1,y_set_2),axis = 0)
y4 = np.concatenate((y4,y_set_3),axis=0)
y4 = np.concatenate((y4,y_set_5),axis=0)


# In[577]:


#Keeping set_5 as validation set
#Concatenating all the remaining four sets of input features and y
x5 = np.concatenate((input_features_set_1,input_features_set_2),axis = 0)
x5 = np.concatenate((x5,input_features_set_3),axis = 0)
x5 = np.concatenate((x5,input_features_set_4),axis = 0)
y5 = np.concatenate((y_set_1,y_set_2),axis = 0)
y5 = np.concatenate((y5,y_set_3),axis=0)
y5 = np.concatenate((y5,y_set_4),axis=0)


# In[578]:


training_error_list = []
validation_error_list = []


# In[579]:


w1,training_1,validation_1 = gradient_descent(x1,y1,input_features_set_1,y_set_1)
print('Training errors ')
print(len(training_1))
print(training_1)
print('Validation errors ')
print(len(validation_1))
print(validation_1)


# In[580]:


w2,training_2,validation_2 = gradient_descent(x2,y2,input_features_set_2,y_set_2)
print('Training errors ')
print(len(training_2))
print(training_2)
print('Validation errors ')
print(len(validation_2))
print(validation_2)


# In[581]:


w3,training_3,validation_3 = gradient_descent(x3,y3,input_features_set_3,y_set_3)
print('Training errors ')
print(len(training_3))
print(training_3)
print('Validation errors ')
print(len(validation_3))
print(validation_3)


# In[582]:


w4,training_4,validation_4 = gradient_descent(x4,y4,input_features_set_4,y_set_4)
print('Training errors ')
print(len(training_4))
print(training_4)
print('Validation errors ')
print(len(validation_4))
print(validation_4)


# In[583]:


w5,training_5,validation_5 = gradient_descent(x5,y5,input_features_set_5,y_set_5)
print('Training errors ')
print(len(training_5))
print(training_5)
print('Validation errors ')
print(len(validation_5))
print(validation_5)


# In[584]:


min1 = min(len(training_1),len(training_2),len(training_3),len(training_4),len(training_5))
print(min1)


# In[585]:


for i in range(min1):
    training_error_list.append((training_1[i]+training_2[i]+training_3[i]+training_4[i]+training_5[i])/5)
    validation_error_list.append((validation_1[i]+validation_2[i]+validation_3[i]+validation_4[i]+validation_5[i])/5)
print(len(training_error_list))
print(training_error_list)
print(len(validation_error_list))
print(validation_error_list)


# In[587]:


from matplotlib import pyplot as plt
plt.plot(training_error_list,label= 'training_error')
plt.plot(validation_error_list,label = 'validation_error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.grid()
plt.legend(loc='upper right')
plt.show()


# In[ ]:




