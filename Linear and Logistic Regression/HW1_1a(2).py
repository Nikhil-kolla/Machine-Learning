#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('D:\\DataSets\\abalone\\abalone\\Dataset.Data',delimiter=" ")


# In[202]:


print(df.shape)


# In[203]:


print(df.columns)


# In[204]:


df['sex'] = pd.factorize(df['sex'])[0]
#print(df)


# In[205]:


input_features=df.drop(['rings'],axis=1)
print(input_features.shape)
ones = np.ones([len(df),1])
print(ones.shape)
#input_features = pd.concat((ones,input_features),axis=1)
input_features_list = [pd.DataFrame(ones),input_features]
input_features = pd.concat(input_features_list,axis=1)
#input_features = input_features.rename(columns = {"0": "Ones"})
print(input_features.shape)
#print(input_features)
y=df['rings']
print(y.shape)


# In[206]:


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


# In[207]:


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


# In[208]:


def normal_equation(x,y):
    temp = np.matmul(x.transpose(),x)
    temp = np.linalg.inv(temp)
    temp = np.matmul(temp,x.transpose())
    temp = np.matmul(temp,y)
    return temp


# In[209]:


def rmse(x,y,w):
    temp = np.dot(x,w)
    temp = np.subtract(y,temp)
    temp = temp**2
    answer = np.sum(temp)
    answer = np.divide(answer,len(x))
    answer = np.sqrt(answer)
    return answer
    


# In[210]:


training_errors = []
validation_errors = []


# In[211]:


#Keeping set_1 as validation set
#Concatenating all the remaining four sets of input features and y
x = np.concatenate((input_features_set_2,input_features_set_3),axis = 0)
x = np.concatenate((x,input_features_set_4),axis = 0)
x = np.concatenate((x,input_features_set_5),axis = 0)
y = np.concatenate((y_set_2,y_set_3),axis = 0)
y = np.concatenate((y,y_set_4),axis=0)
y = np.concatenate((y,y_set_5),axis=0)
optimal_w_set_1 = normal_equation(x,y)
#print(optimal_w_set_1)
#print('The training error is ')
training_errors.append(rmse(x,y,optimal_w_set_1))
#print('The validation error is ')
validation_errors.append(rmse(input_features_set_1,y_set_1,optimal_w_set_1))


# In[212]:


#Keeping set_2 as validation set
#Concatenating all the remaining four sets of input features and y
x = np.concatenate((input_features_set_1,input_features_set_3),axis = 0)
x = np.concatenate((x,input_features_set_4),axis = 0)
x = np.concatenate((x,input_features_set_5),axis = 0)
y = np.concatenate((y_set_1,y_set_3),axis = 0)
y = np.concatenate((y,y_set_4),axis=0)
y = np.concatenate((y,y_set_5),axis=0)
optimal_w_set_2 = normal_equation(x,y)
#print(optimal_w_set_2)
#print('The training error is ')
training_errors.append(rmse(x,y,optimal_w_set_2))
#print('The validation error is ')
validation_errors.append(rmse(input_features_set_2,y_set_2,optimal_w_set_2))


# In[213]:


#Keeping set_3 as validation set
#Concatenating all the remaining four sets of input features and y
x = np.concatenate((input_features_set_1,input_features_set_2),axis = 0)
x = np.concatenate((x,input_features_set_4),axis = 0)
x = np.concatenate((x,input_features_set_5),axis = 0)
y = np.concatenate((y_set_1,y_set_2),axis = 0)
y = np.concatenate((y,y_set_4),axis=0)
y = np.concatenate((y,y_set_5),axis=0)
optimal_w_set_3 = normal_equation(x,y)
#print(optimal_w_set_3)
#print('The training error is ')
training_errors.append(rmse(x,y,optimal_w_set_3))
#print('The validation error is ')
validation_errors.append(rmse(input_features_set_3,y_set_3,optimal_w_set_3))


# In[214]:


#Keeping set_4 as validation set
#Concatenating all the remaining four sets of input features and y
x = np.concatenate((input_features_set_1,input_features_set_2),axis = 0)
x = np.concatenate((x,input_features_set_3),axis = 0)
x = np.concatenate((x,input_features_set_5),axis = 0)
y = np.concatenate((y_set_1,y_set_2),axis = 0)
y = np.concatenate((y,y_set_3),axis=0)
y = np.concatenate((y,y_set_5),axis=0)
optimal_w_set_4 = normal_equation(x,y)
#print(optimal_w_set_4)
#print('The training error is ')
training_errors.append(rmse(x,y,optimal_w_set_4))
#print('The validation error is ')
validation_errors.append(rmse(input_features_set_4,y_set_4,optimal_w_set_4))


# In[215]:


#Keeping set_5 as validation set
#Concatenating all the remaining four sets of input features and y
x = np.concatenate((input_features_set_1,input_features_set_2),axis = 0)
x = np.concatenate((x,input_features_set_3),axis = 0)
x = np.concatenate((x,input_features_set_4),axis = 0)
y = np.concatenate((y_set_1,y_set_2),axis = 0)
y = np.concatenate((y,y_set_3),axis=0)
y = np.concatenate((y,y_set_4),axis=0)
optimal_w_set_5 = normal_equation(x,y)
#print(optimal_w_set_5)
#print('The training error is ')
training_errors.append(rmse(x,y,optimal_w_set_5))
#print('The validation error is ')
validation_errors.append(rmse(input_features_set_5,y_set_5,optimal_w_set_5))


# In[216]:


print('The training errors ')
print(training_errors)
print('The Validation errors ')
print(validation_errors)


# In[ ]:





# In[ ]:




