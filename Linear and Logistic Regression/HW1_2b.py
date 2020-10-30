#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import idx2numpy
from sklearn.linear_model import LogisticRegression


# In[30]:


file1 = 'D:\\DataSets\\MNIST\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
x_train = idx2numpy.convert_from_file(file1)
x_train = x_train.reshape(60000,28*28)


# In[31]:


#file2 = 'D:\\DataSets\\MNIST\\train-labels-idx1-ubyte\\train-labels'
y_train = idx2numpy.convert_from_file('D:\\DataSets\\MNIST\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte') 


# In[32]:


file2 = 'D:\\DataSets\\MNIST\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
x_test = idx2numpy.convert_from_file(file2)


# In[33]:


x_test =x_test.reshape(10000,28*28)


# In[34]:


y_test = idx2numpy.convert_from_file('D:\\DataSets\\MNIST\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')


# In[35]:


model1 = LogisticRegression(penalty='l1',solver='saga',max_iter=1,multi_class='ovr').fit(x_train, y_train)


# In[36]:


model2 = LogisticRegression(penalty='l2',solver='saga',max_iter=1,multi_class='ovr').fit(x_train,y_train)


# In[37]:


print('Training accuracy using L1: ',model1.score(x_train,y_train)*100)
print('Training accuracy using L2: ',model2.score(x_train,y_train)*100)
print('Testing accuracy using L1: ',model1.score(x_test,y_test)*100)
print('Testing accuracy using L2: ',model2.score(x_test,y_test)*100)


# In[ ]:




