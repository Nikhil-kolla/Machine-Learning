#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('D:\\DataSets\\abalone\\abalone\\Dataset.Data',delimiter=" ")


# In[89]:


print(df.shape)


# In[90]:


print(df.columns)


# In[91]:


df['sex'] = pd.factorize(df['sex'])[0]


# In[92]:


input_features=df.drop(['rings'],axis=1)
print(input_features.shape)
ones = np.ones([len(df),1])
print(ones.shape)
input_features_list = [pd.DataFrame(ones),input_features]
input_features = pd.concat(input_features_list,axis=1)
print(input_features.shape)
y=df['rings']
print(y.shape)


# In[93]:


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


# In[94]:


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


# In[95]:


x2 = np.concatenate((input_features_set_1,input_features_set_3),axis = 0)
x2 = np.concatenate((x2,input_features_set_4),axis = 0)
x2 = np.concatenate((x2,input_features_set_5),axis = 0)
y2 = np.concatenate((y_set_1,y_set_3),axis = 0)
y2 = np.concatenate((y2,y_set_4),axis=0)
y2 = np.concatenate((y2,y_set_5),axis=0)


# # L2 REGULARIZATION

# In[96]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
l2_value={'alpha':(0.00000001,0.01,0.001,1,0.1,0.0001,0,0.00001)}
#l2_value={'alpha':(25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01)}
r_estimator = Ridge(alpha=1)
#model = Ridge()
grid = GridSearchCV(estimator=r_estimator, param_grid=l2_value,cv=5)
grid.fit(x2,y2)
best_l2=grid.best_params_
best_l2=best_l2['alpha']
print(best_l2)


# In[97]:


def rmse(x,y,w):
    temp = x.dot(w)
    temp = np.subtract(y,temp)
    temp = temp**2
    answer = np.sum(temp)
    answer = np.divide(answer,len(x))
    answer = np.sqrt(answer)
    return answer


# In[98]:


def cost(x,y,w):
    total_loss = 0
    for i in range(0,len(x)):
        loss = y[i]-x[i].dot(w)
        loss = -2 * loss
        loss = loss * x[i]
        total_loss += loss
    temp = float(1/best_l2)*w.dot(w.T)
    total_loss = total_loss+temp
    return total_loss


# In[99]:


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


# In[100]:


w2,training_2,validation_2 = gradient_descent(x2,y2,input_features_set_2,y_set_2)
print('Training errors ')
print(len(training_2))
print(training_2)
print('Validation errors ')
print(len(validation_2))
print(validation_2)


# In[101]:


from matplotlib import pyplot as plt
plt.plot(training_2,label= 'training_error')
plt.plot(validation_2,label = 'validation_error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('L2 Regularization')
plt.grid()
plt.legend(loc='upper right')
plt.show()


# # L1 REGULARIZATION

# In[102]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
l2_value={'alpha':(0.00000001,0.01,0.001,1,0.1,0.0001,0,0.00001)}
#l2_value={'alpha':(25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01)}
r_estimator = Lasso(alpha=1)
#model = Ridge()
grid = GridSearchCV(estimator=r_estimator, param_grid=l2_value,cv=5)
grid.fit(x2,y2)
best_l1=grid.best_params_
best_l1=best_l1['alpha']
print(best_l1)


# In[103]:


def cost(x,y,w):
    total_loss = 0
    for i in range(0,len(x)):
        loss = y[i]-x[i].dot(w)
        loss = -2 * loss
        loss = loss * x[i]
        total_loss += loss
    temp = (1/best_l2)*w
    total_loss = total_loss+temp
    return total_loss


# In[104]:


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


# In[105]:


w2,training_2,validation_2 = gradient_descent(x2,y2,input_features_set_2,y_set_2)
print('Training errors ')
print(len(training_2))
print(training_2)
print('Validation errors ')
print(len(validation_2))
print(validation_2)


# In[106]:


from matplotlib import pyplot as plt
plt.plot(training_2,label= 'training_error')
plt.plot(validation_2,label = 'validation_error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('L1 Regularization')
plt.grid()
plt.legend(loc='upper right')
plt.show()


# In[ ]:




