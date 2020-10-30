#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#reference
#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60


# In[2]:


from sklearn import datasets
from sklearn.svm import SVC 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as trainTestSplit
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as py
from sklearn.metrics import f1_score,accuracy_score,roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import roc_curve
import pickle
from sklearn.decomposition import PCA


# In[3]:


#reference
#https://stackoverflow.com/questions/53090114/concatenate-5-unpickle-dictionaries-without-overwriting-data-is-from-cifar-10


# In[4]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[5]:


dict1 = unpickle("C:\\Users\\hanum\\Desktop\\DataSets\\cifar-10-batches-py\\data_batch_1")
dict2 = unpickle("C:\\Users\\hanum\\Desktop\\DataSets\\cifar-10-batches-py\\data_batch_2")
dict3 = unpickle("C:\\Users\\hanum\\Desktop\\DataSets\\cifar-10-batches-py\\data_batch_3")
dict4 = unpickle("C:\\Users\\hanum\\Desktop\\DataSets\\cifar-10-batches-py\\data_batch_4")
dict5 = unpickle("C:\\Users\\hanum\\Desktop\\DataSets\\cifar-10-batches-py\\data_batch_5")


# In[6]:


type(dict1)


# In[7]:


#reference
#https://stackoverflow.com/questions/42466639/convert-a-dictionary-to-a-pandas-dataframe


# In[8]:


def dictionaryToDataframe(di):
    df={'data':list(di[b'data']),'data_label':list(di[b'labels'])}
    df=pd.DataFrame(df)
    return df


# In[9]:


data1 = dictionaryToDataframe(dict1)
data2 = dictionaryToDataframe(dict2)
data3 = dictionaryToDataframe(dict3)
data4 = dictionaryToDataframe(dict4)
data5 = dictionaryToDataframe(dict5)


# In[10]:


test_dict = unpickle("C:\\Users\\hanum\\Desktop\\DataSets\\cifar-10-batches-py\\test_batch")


# In[11]:


test_data = dictionaryToDataframe(test_dict)


# In[12]:


#Concatenating the training data frames
train_data = np.concatenate((data1,data2),axis=0)
train_data = np.concatenate((train_data,data3),axis=0)
train_data = np.concatenate((train_data,data4),axis=0)
train_data = np.concatenate((train_data,data5),axis=0)


# In[13]:


print(train_data.shape)


# In[14]:


print(test_data.shape)


# In[15]:


a = train_data
b = test_data


# In[16]:


def p2c(d1):
    a=[]
    for i in range(len(d1)):
        b = d1.iloc[i,0]
        a.append(b)
    return np.array(a)


# In[17]:


def getlabel(d1):
    a=[]
    for i in range(len(d1)):
        b = d1.iloc[i,1]
        a.append(b)
    return np.array(a)


# In[18]:


train_input = p2c(pd.DataFrame(train_data))
print(train_input.shape)


# In[19]:


test_input = p2c(pd.DataFrame(test_data))
print(train_input.shape)


# In[20]:


from sklearn import preprocessing
mmscale = preprocessing.MinMaxScaler(feature_range=(0,1))
train_input = mmscale.fit_transform(train_input)
test_input = mmscale.fit_transform(test_input)


# In[21]:


print(train_input.shape)
print(test_input.shape)


# In[22]:


start_time = time.time()
pca = PCA(.95)
pca.fit(train_input)
end_time = time.time()-start_time
print("Time taken for PCA with 95% of data preserving is",end_time/60)


# In[23]:


train_input = pca.transform(train_input)
print(train_input.shape)


# In[24]:


test_input = pca.transform(test_input)
print(test_input.shape)


# In[99]:


# train_df=pd.DataFrame(train_data)
# test_df=pd.DataFrame(test_data)


# In[100]:


# train_y = pd.DataFrame(data=train_df['data_label'].values,columns=['label'])


# In[25]:


train_labels = getlabel(pd.DataFrame(a))


# In[26]:


print(train_labels.shape)


# In[27]:


test_labels = getlabel(pd.DataFrame(b))
print(test_labels.shape)


# In[28]:


print(train_input.shape)
print(test_input.shape)
print(train_labels.shape)
print(test_labels.shape)


# In[29]:


print(type(train_input))
print(type(test_input))
print(type(train_labels))
print(type(test_labels))


# In[30]:


sample_input,left_input,sample_labels,left_labels = trainTestSplit(train_input,train_labels,train_size=0.20,stratify=train_labels)


# In[31]:


print(sample_input.shape)
print(left_input.shape)
print(sample_labels.shape)
print(left_labels.shape)


# In[110]:


parameters = [{'kernel':['linear','rbf'],'gamma':[10,1,1e-1,1e-2],'C':[1e-2,1e-1,1,10]}]
start_time = time.time()
clf = GridSearchCV(SVC(),parameters,cv=3)
clf.fit(sample_input,sample_labels)
end_time = time.time()-start_time
print("Time taken for GridSearchCV is",end_time/60)


# In[34]:


print(type(train_input))
print(type(train_labels))


# In[32]:


print("Best scores:",clf.best_score_)
print("Best params:",clf.best_params_)


# In[36]:


start_time = time.time()
modelSVC = SVC(kernel='rbf',C=10,gamma=0.01)
modelSVC.fit(train_input,train_labels)
end_time = time.time()-start_time
print("Time taken for training is",end_time/60)


# In[56]:


start_time = time.time()
print("The train accuracy is")
print(modelSVC.score(train_input,train_labels))
print("The test accuracy is")
print(modelSVC.score(test_input,test_labels))
end_time = time.time()-start_time
print("Time taken for testing is",end_time/60)


# # Getting Support Vectors

# In[ ]:


#reference
#https://scikit-learn.org/stable/modules/svm.html


# In[57]:


support_index=modelSVC.support_
print(support_index.shape)


# In[58]:


input_support = train_input[support_index,:]


# In[59]:


print(input_support.shape)


# In[60]:


labels_support = train_labels[support_index]


# In[61]:


print(labels_support.shape)


# In[62]:


start_time = time.time()
modelSVC2 = SVC(kernel='rbf',C=10,gamma=0.01)
modelSVC2.fit(input_support,labels_support)
end_time = time.time()-start_time
print("Time taken for training is",end_time/60)


# In[63]:


start_time = time.time()
print("The train accuracy is")
print(modelSVC2.score(train_input,train_labels))
print("The test accuracy is")
print(modelSVC2.score(test_input,test_labels))
end_time = time.time()-start_time
print("Time taken for testing is",end_time/60)


# In[ ]:




