#!/usr/bin/env python
# coding: utf-8

# # Assignment-2:Question-2

# In[101]:


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


# In[3]:


data_array = datasets.load_wine()


# In[4]:


data=pd.DataFrame(data_array.data,columns=data_array.feature_names)


# In[5]:


data['target_values']=pd.Series(data_array.target)


# In[6]:


print(data.head())


# In[7]:


data.shape


# In[8]:


input_features=data.drop(['target_values'],axis=1)
labels=data['target_values']


# In[9]:


input_features.shape


# In[ ]:


#reference
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html


# In[140]:


from sklearn import preprocessing
mmscale = preprocessing.MinMaxScaler(feature_range=(0,1))
input_features = mmscale.fit_transform(input_features)
input_features


# In[11]:


labels.shape


# In[ ]:


#referenec
#https://seaborn.pydata.org/generated/seaborn.pairplot.html


# In[432]:


sb.pairplot(data,hue='target_values')


# In[12]:


train_input_features,test_input_features=trainTestSplit(input_features,test_size=0.3,random_state=42)
train_labels,test_labels=trainTestSplit(labels,test_size=0.3,random_state=42)


# In[13]:


print(train_input_features.shape)
print(test_input_features.shape)
print(train_labels.shape)
print(test_labels.shape)


# In[ ]:


#reference
#https://stackoverflow.com/questions/36285155/pandas-get-dummies


# In[14]:


train_labels_dummies=pd.get_dummies(train_labels)
test_labels_dummies=pd.get_dummies(test_labels)


# In[15]:


train_labels_dummies=train_labels_dummies.to_numpy()
test_labels_dummies=test_labels_dummies.to_numpy()


# In[16]:


print(type(train_input_features))
print(type(train_labels_dummies))
print(type(test_input_features))
print(type(test_labels_dummies))


# In[ ]:


#reference
#https://datastoriesweb.wordpress.com/2017/06/11/classification-one-vs-rest-and-one-vs-one/


# # SVM-One vs. Rest

# In[266]:


start_time=time.time()
model0=SVC(kernel='linear')
model0.fit(train_input_features,train_labels_dummies[:,[0]])
b0=model0.intercept_
weights0=model0.coef_
end_time=(time.time()-start_time)*1000
print(f'Training time of model-OVR-class-0: {"{0:.4f}".format(end_time)} ms')


# In[433]:


print("The weight matrix for SVM-OVR classifier with class label 0")
print(weights0)
print("The intercept for SVM-OVR classifier with class label 0")
print(b0)


# In[19]:


train_predicted0=train_input_features.dot(weights0.transpose())+b0
test_predicted0=test_input_features.dot(weights0.transpose())+b0


# In[20]:


print(train_predicted0.shape)


# In[440]:


start_time=time.time()
model1=SVC(kernel='linear')
model1.fit(train_input_features,train_labels_dummies[:,[1]])
b1=model1.intercept_
weights1=model1.coef_
end_time=(time.time()-start_time)*1000
print(f'Training time of model-OVR-class-1: {"{0:.4f}".format(end_time)} ms')


# In[441]:


print("The weight matrix for SVM-OVR classifier with class label 1")
print(weights1)
print("The intercept for SVM-OVR classifier with class label 1")
print(b1)


# In[442]:


train_predicted1=train_input_features.dot(weights1.transpose())+b1
test_predicted1=test_input_features.dot(weights1.transpose())+b1


# In[443]:


print(train_predicted1.shape)
print(test_predicted1.shape)


# In[444]:


start_time=time.time()
model2=SVC(kernel='linear')
model2.fit(train_input_features,train_labels_dummies[:,[2]])
b2=model2.intercept_
weights2=model2.coef_
end_time=(time.time()-start_time)*1000
print(f'Training time of model-OVR-class-2: {"{0:.4f}".format(end_time)} ms')


# In[437]:


print("The weight matrix for SVM-OVR classifier with class label 2")
print(weights2)
print("The intercept for SVM-OVR classifier with class label 2")
print(b2)


# In[27]:


train_predicted2=train_input_features.dot(weights2.transpose())+b2
test_predicted2=test_input_features.dot(weights2.transpose())+b2


# In[28]:


print(train_predicted2.shape)
print(test_predicted2.shape)


# In[29]:


def calculateAccuracy(predicted_values,true_values,value):
    c=0
    for i in range(len(predicted_values)):
        if (true_values[i]==value and predicted_values[i]>=0):
            c+=1
        elif (true_values[i]!=value and predicted_values[i]<0):
            c+=1
        else:
            c+=0
    return (c/len(true_values))*100


# In[30]:


train_labels=train_labels.to_numpy()
test_labels=test_labels.to_numpy()


# In[31]:


print((type(train_predicted0)))
print(type(train_labels))
print(type(test_labels))


# # Training Accuracy for 3 Classes

# In[32]:


training_accuracy0=calculateAccuracy(train_predicted0,train_labels,0)
training_accuracy1=calculateAccuracy(train_predicted1,train_labels,1)
training_accuracy2=calculateAccuracy(train_predicted2,train_labels,2)


# In[142]:


print("The training accuracies of 3 classes: ")
print("Training accuracy for class-0 ",training_accuracy0)
print("Training accuracy for class-1 ",training_accuracy1)
print("Training accuracy for class-2 ",training_accuracy2)


# # Test Accuracy for 3 Classes

# In[34]:


test_accuracy0=calculateAccuracy(test_predicted0,test_labels,0)
test_accuracy1=calculateAccuracy(test_predicted1,test_labels,1)
test_accuracy2=calculateAccuracy(test_predicted2,test_labels,2)


# In[144]:


print("The test accuracies of 3 classes is: ")
print("Test accuracy for class-0 ",test_accuracy0)
print("Test accuracy for class-1 ",test_accuracy1)
print("Test accuracy for class-2 ",test_accuracy2)


# In[145]:


test_labels=test_labels.reshape(54,1)


# In[146]:


print(test_predicted0.shape)
print(test_predicted1.shape)
print(test_predicted2.shape)
print(test_labels.shape)


# In[147]:


print(type(test_labels))


# In[38]:


def calculateTotalAccuracy(predicted0,predicted1,predicted2,true_values):
    rows=len(true_values)
    c=0
    finalPrediction=true_values*0
    for i in range(rows):
        if(true_values[i]==0 and predicted0[i]>0 and predicted1[i]<0 and predicted2[i]<0):
            c+=1
            finalPrediction[i]=0
        elif(true_values[i]==1 and predicted0[i]<0 and predicted1[i]>0 and predicted2[i]<0):
            c+=1
            finalPrediction[i]=1
        elif(true_values[i]==2 and predicted0[i]<0 and predicted1[i]<0 and predicted2[i]>0):
            c+=1
            finalPrediction[i]=2
        else:
            c+=0
    accuracy = (c/rows)*100
    return accuracy,finalPrediction


# # TrainAccuracy

# In[151]:


totalAccuracy_train,finalPrediction_train=calculateTotalAccuracy(train_predicted0,train_predicted1,train_predicted2,train_labels)


# In[152]:


print("The total training accuracy for SVM-OVR is: ")
print(totalAccuracy_train)


# # TestAccuracy

# In[153]:


totalAccuracy_test,finalPrediction_test=calculateTotalAccuracy(test_predicted0,test_predicted1,test_predicted2,test_labels)


# In[154]:


print("The total test accuracy for SVM-OVR is: ")
print(totalAccuracy_test)


# # F1 Score

# In[155]:


print("The F1 Score for SVM-OVR is: ")
f1_score(test_labels,finalPrediction_test,average=None)


# # Accuracy Score

# In[156]:


print("The accuracy score for SVM-OVR is: ")
accuracy_score(test_labels,finalPrediction_test)


# # ROC Curve

# In[438]:


FalsePositive0,TruePositive0,threshold=roc_curve(test_labels_dummies[:,[0]],test_predicted0)
py.plot(FalsePositive0,TruePositive0,label="Class 0")
FalsePositive1,TruePositive1,threshold=roc_curve(test_labels_dummies[:,[1]],test_predicted1)
py.plot(FalsePositive1,TruePositive1,label="Class 1")
FalsePositive2,TruePositive2,threshold=roc_curve(test_labels_dummies[:,[2]],test_predicted2)
py.plot(FalsePositive2,TruePositive2,label="Class 2")
py.title("ROC CURVE-SVM-OVR")
py.xlabel("False Positive Rate")
py.ylabel("True Positive Rate")
py.legend()
py.show()


# # SVM One vs One

# In[ ]:


#reference
# https://datastoriesweb.wordpress.com/2017/06/11/classification-one-vs-rest-and-one-vs-one/


# In[369]:


def predict_OVO(predicted_probs,val1,val2):
    values=[]
    for i in range(len(predicted_probs)):
        if(predicted_probs[i]>0):
            values.append(val1)
        else:
            values.append(val2)
    return np.array(values)


# In[370]:


def max3(a,b,c):
    if(a>b and a>c):
        return 0
    if(b>a and b>c):
        return 1
    if(c>a and c>b):
        return 2


# In[371]:


def final_assign(list1,list2,list3):
    answer=[]
    for i in range(len(list1)):
        c0=c1=c2=0
        if(list1[i]==0):
            c0=c0+1
        if(list2[i]==0):
            c0=c0+1
        if(list3[i]==0):
            c0=c0+1
        if(list1[i]==1):
            c1=c1+1
        if(list2[i]==1):
            c1=c1+1
        if(list3[i]==1):
            c1=c1+1
        if(list1[i]==2):
            c2=c2+1
        if(list2[i]==2):
            c2=c2+1
        if(list3[i]==2):
            c2=c2+1
        answer.append(max3(c0,c1,c2))
    return np.array(answer)  


# In[372]:


def createOnevsOneSet(x,y,val1,val2):
    new_x=[]
    new_y=[]
    for i in range(len(y)):
        if(y[i]==val1 or y[i]==val2):
            new_x.append(x[i])
            if(y[i]==val1):
                new_y.append(1)
            if(y[i]==val2):
                new_y.append(0)
    new_x=np.array(new_x)
    new_y=np.array(new_y)
    return new_x,new_y


# In[402]:


train_features01,train_labels01=createOnevsOneSet(train_input_features,train_labels,0,1)
train_features02,train_labels02=createOnevsOneSet(train_input_features,train_labels,0,2)
train_features12,train_labels12=createOnevsOneSet(train_input_features,train_labels,1,2)


# In[403]:


print(train_features01.shape)
print(train_labels01.shape)
print(train_features02.shape)
print(train_labels02.shape)
print(train_features12.shape)
print(train_labels12.shape)


# In[449]:


start_time=time.time()
model_OVO_01=SVC(kernel='linear')
model_OVO_01.fit(train_features01,train_labels01)
b_OVO_01=model_OVO_01.intercept_
weights_OVO_01=model_OVO_01.coef_
print("The weight matrix for SVM-OVO classifier of model01")
print(weights_OVO_01)
print("The intercept for SVM-OVO classifier of model01")
print(b_OVO_01)
test_predicted_OVO_01=test_input_features.dot(weights_OVO_01.transpose())+b_OVO_01
test_predicted_OVO_01=predict_OVO(test_predicted_OVO_01,0,1)
test_predicted_OVO_01=test_predicted_OVO_01.transpose()
end_time=(time.time()-start_time)*1000
print(f'Training time of model-OVO-classifier-01: {"{0:.4f}".format(end_time)} ms')


# In[450]:


start_time=time.time()
model_OVO_02=SVC(kernel='linear')
model_OVO_02.fit(train_features02,train_labels02)
b_OVO_02=model_OVO_02.intercept_
weights_OVO_02=model_OVO_02.coef_
print("The weight matrix for SVM-OVO classifier of model02")
print(weights_OVO_02)
print("The intercept for SVM-OVO classifier of model02")
print(b_OVO_02)
test_predicted_OVO_02=test_input_features.dot(weights_OVO_02.transpose())+b_OVO_02
test_predicted_OVO_02=predict_OVO(test_predicted_OVO_02,0,2)
test_predicted_OVO_02=test_predicted_OVO_02.transpose()
end_time=(time.time()-start_time)*1000
print(f'Training time of model-OVO-classifier-02: {"{0:.4f}".format(end_time)} ms')


# In[452]:


start_time=time.time()
model_OVO_12=SVC(kernel='linear')
model_OVO_12.fit(train_features12,train_labels12)
b_OVO_12=model_OVO_12.intercept_
weights_OVO_12=model_OVO_12.coef_
print("The weight matrix for SVM-OVO classifier of model12")
print(weights_OVO_12)
print("The intercept for SVM-OVO classifier of model12")
print(b_OVO_12)
test_predicted_OVO_12=test_input_features.dot(weights_OVO_12.transpose())+b_OVO_12
test_predicted_OVO_12=predict_OVO(test_predicted_OVO_12,1,2)
test_predicted_OVO_12=test_predicted_OVO_12.transpose()
end_time=(time.time()-start_time)*1000
print(f'Training time of model-OVO-classifier-12: {"{0:.4f}".format(end_time)} ms')


# In[407]:


print(test_predicted_OVO_01.shape)
print(test_predicted_OVO_02.shape)
print(test_predicted_OVO_12.shape)
print(test_labels.shape)


# In[408]:


print(test_predicted_OVO_01)


# In[427]:


print(test_predicted_OVO_02)


# In[410]:


print(test_predicted_OVO_12)


# In[411]:


predicted_answer=final_assign(test_predicted_OVO_01,test_predicted_OVO_02,test_predicted_OVO_12)


# In[412]:


print(predicted_answer.transpose())


# In[413]:


print(predicted_answer.shape)


# In[414]:


print(test_labels.transpose())


# In[415]:


print(test_labels.shape)


# In[416]:


def calculateAccuracyClassWise(true,pred,val):
    c=0
    predicted=[]
    for i in range(len(true)):
        if (true[i]==pred[i] and pred[i]==val):
            c+=1
            predicted.append(1)
        elif (true[i]!=val and pred[i]!=val):
            c+=1
            predicted.append(0)
        else:
            c+=0
            predicted.append(0)
    return (c/len(true))*100,predicted


# In[417]:


def calculateTotalAccuracy_OVO(true_values,predicted):
    finalPrediction=[]
    c=0
    for i in range(len(predicted)):
        if(true_values[i]==predicted[i] and predicted[i]==0):
            c+=1
            finalPrediction.append(0)
        elif(true_values[i]==predicted[i] and predicted[i]==1):
            c+=1
            finalPrediction.append(1)
        elif(true_values[i]==predicted[i] and predicted[i]==2):
            c+=1
            finalPrediction.append(2)
        else:
            c+=0
            #finalPrediction.append(3)
    accuracy = (c/len(predicted))*100
    return accuracy,finalPrediction


# In[418]:


totalAccuracy_OVO,finalPrediction_OVO=calculateTotalAccuracy_OVO(test_labels,predicted_answer)


# In[454]:


print("The total accuracy of SVM-OVO on test data")
print(totalAccuracy_OVO)


# In[420]:


test_accuracy_OVO_0,predicted_OVO_0=calculateAccuracyClassWise(predicted_answer,test_labels,0)
test_accuracy_OVO_1,predicted_OVO_1=calculateAccuracyClassWise(predicted_answer,test_labels,1)
test_accuracy_OVO_2,predicted_OVO_2=calculateAccuracyClassWise(predicted_answer,test_labels,2)


# In[455]:


print("The test accuracies of 3 classes is: ")
print("For class 0")
print(test_accuracy_OVO_0)
print("For class 1")
print(test_accuracy_OVO_1)
print("For class 2")
print(test_accuracy_OVO_2)


# # F1 Score: One vs. One

# In[425]:


print("The F1 Score for SVM-OVO is: ")
f1_score(test_labels,finalPrediction_OVO,average=None)


# # Accuracy Score: One vs. One

# In[426]:


print("The accuracy score for SVM-OVO is: ")
accuracy_score(test_labels,finalPrediction_OVO)


# # ROC_Curve: Ove vs. One

# In[439]:


FalsePositive_OVO_0,TruePositive_OVO_0,threshold=roc_curve(test_labels_dummies[:,[0]],predicted_OVO_0)
py.plot(FalsePositive_OVO_0,TruePositive_OVO_0,label="Class 0")
FalsePositive_OVO_1,TruePositive_OVO_1,threshold=roc_curve(test_labels_dummies[:,[1]],predicted_OVO_1)
py.plot(FalsePositive_OVO_1,TruePositive_OVO_1,label="Class 1")
FalsePositive_OVO_2,TruePositive_OVO_2,threshold=roc_curve(test_labels_dummies[:,[2]],predicted_OVO_2)
py.plot(FalsePositive_OVO_2,TruePositive_OVO_2,label="Class 2")
py.title("ROC CURVE-SVM-OVO")
py.xlabel("False Positive Rate")
py.ylabel("True Positive Rate")
py.legend()
py.show()


# # Gaussian Naive Bayes

# In[158]:


def calculateAccuracyGNB(predicted,truth):
    c=0
    for i in range(len(truth)):
        if(predicted[i]==truth[i]):
            c+=1
        else:
            c+=0
    return (c/(len(truth)))*100


# In[159]:


def calculateAccuracyGNB_label(predicted,truth,value):
    c=0
    for i in range(len(truth)):
        if(predicted[i]==value and truth[i]==value):
            c+=1
        elif(predicted[i]!=value and truth[i]!=value):
            c+=1
        else:
            c+=0
    return (c/(len(truth)))*100


# In[160]:


def convert(actual,value):
    temp = actual.copy()
    for i in range(len(actual)):
        if(actual[i]==value):
            temp[i]=1
        else:
            temp[i]=0
    return temp


# In[269]:


start_time=time.time()
modelGNB=GaussianNB()
modelGNB.fit(train_input_features,train_labels)
predictedGNB=modelGNB.predict(test_input_features)
predictedGNB_proba=modelGNB.predict_proba(train_input_features)
end_time=(time.time()-start_time)*1000
print(f'Run time of Gaussian Naive Bayes model is: {"{0:.4f}".format(end_time)} ms')


# In[270]:


accuracy_GNB=calculateAccuracyGNB(predictedGNB,test_labels)
print("Total Accuracy for Gaussian Naive Bayes on Test Data is: ")
print(accuracy_GNB)


# # Test Accuracies

# In[271]:


test_accuracy_gnb0=calculateAccuracyGNB_label(predictedGNB,test_labels,0)
test_accuracy_gnb1=calculateAccuracyGNB_label(predictedGNB,test_labels,1)
test_accuracy_gnb2=calculateAccuracyGNB_label(predictedGNB,test_labels,2)
print("Individual Accuracy for class-0 on Test Data is: ")
print(test_accuracy_gnb0)
print("Individual Accuracy for class-1 on Test Data is: ")
print(test_accuracy_gnb1)
print("Individual Accuracy for class-2 on Test Data is: ")
print(test_accuracy_gnb2)


# In[165]:


testgnb0=convert(predictedGNB,0)
testgnb1=convert(predictedGNB,1)
testgnb2=convert(predictedGNB,2)


# # F1-Score

# In[272]:


print("The F1 Score for Guassian Naive Bayes is on Test Data is: ")
f1_score(test_labels,predictedGNB,average=None)


# # Accuracy Score

# In[273]:


print("The accuracy score for Guassian Naive Bayes is on Test Data is: ")
accuracy_score(test_labels,predictedGNB)


# # ROC-Curve Gaussian Naive Bayes

# In[168]:


FalsePositiveGNB0,TruePositiveGNB0,threshold=roc_curve(test_labels_dummies[:,[0]],testgnb0)
py.plot(FalsePositiveGNB0,TruePositiveGNB0,label="Class 0")
FalsePositiveGNB1,TruePositiveGNB1,threshold=roc_curve(test_labels_dummies[:,[1]],testgnb1)
py.plot(FalsePositiveGNB1,TruePositiveGNB1,label="Class 1")
FalsePositiveGNB2,TruePositiveGNB2,threshold=roc_curve(test_labels_dummies[:,[2]],testgnb2)
py.plot(FalsePositiveGNB2,TruePositiveGNB2,label="Class 2")
py.title("ROC CURVE-GNB")
py.xlabel("False Positive Rate")
py.ylabel("True Positive Rate")
py.legend()
py.show()


# # Decision Trees

# In[169]:


print(train_input_features.shape)
print(test_input_features.shape)
print(train_labels.shape)
print(test_labels.shape)


# In[170]:


test_input_DT=train_input_features
test_labels_DT=test_labels
print(test_input_DT.shape)
print(test_labels_DT.shape)


# In[171]:


train_input_DT,val_input_DT,train_labels_DT,val_labels_DT=trainTestSplit(train_input_features,train_labels,train_size=0.8,stratify=train_labels)


# In[172]:


print(train_input_DT.shape)
print(val_input_DT.shape)
print(train_labels_DT.shape)
print(val_labels_DT.shape)


# In[173]:


training_accuracy_DT=[]
validation_accuracy_DT=[]


# In[174]:


depths=range(1,30)


# In[175]:


for depth in depths:
    model_DT = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model_DT.fit(train_input_DT,train_labels_DT)
    training_prediction_DT=model_DT.predict(train_input_DT)
    validation_prediction_DT=model_DT.predict(val_input_DT)
    training_accuracy_DT.append(accuracy_score(train_labels_DT,training_prediction_DT))
    validation_accuracy_DT.append(accuracy_score(val_labels_DT,validation_prediction_DT))


# In[265]:


py.plot(depths,validation_accuracy_DT,label="validation")
py.plot(depths,training_accuracy_DT,label="training")
py.xlabel("Depth of decision tree")
py.ylabel("accuracy scores")
py.legend()
py.show()


# In[230]:


print(training_accuracy_DT)


# In[259]:


model_DT=DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,max_depth=5)
start_time=time.time()
model_DT.fit(train_input_DT,train_labels_DT)
predicted_training_DT=model_DT.predict(train_input_features)
predicted_test_DT=model_DT.predict(test_input_features)
end_time=(time.time()-start_time)*1000
print(f'Run time of the optimal classifier: {"{0:.4f}".format(end_time)} ms')


# In[260]:


predicted_test_prob_DT=model_DT.predict_proba(test_input_features)


# In[261]:


train_accuracy_DT=accuracy_score(train_labels,predicted_training_DT)
test_accuracy_DT=accuracy_score(test_labels,predicted_test_DT)


# In[262]:


def calculateAccuracyDT_label(predicted,truth,value):
    c=0
    for i in range(len(truth)):
        if(predicted[i]==value and truth[i]==value):
            c+=1
        elif(predicted[i]!=value and truth[i]!=value):
            c+=1
        else:
            c+=0
    return (c/(len(truth)))*100


# In[274]:


test_accuracy_DT0=calculateAccuracyDT_label(predicted_test_DT,test_labels,0)
test_accuracy_DT1=calculateAccuracyDT_label(predicted_test_DT,test_labels,1)
test_accuracy_DT2=calculateAccuracyDT_label(predicted_test_DT,test_labels,2)
print("Individual Accuracy for class-0 on Test Data is: ")
print(test_accuracy_DT0)
print("Individual Accuracy for class-1 on Test Data is: ")
print(test_accuracy_DT1)
print("Individual Accuracy for class-2 on Test Data is: ")
print(test_accuracy_DT2)


# In[276]:


print("The training accuracy score is: ",train_accuracy_DT*100,"%")
print("The test accuracy score is: ",test_accuracy_DT*100,"%")


# In[456]:


print("The F1 Score on Test Data is: ")
f1_score(test_labels,predicted_test_DT,average=None)


# In[457]:


print("The accuracy score on Test Data is")
accuracy_score(test_labels,predicted_test_DT)


# In[252]:


def convert(actual,value):
    temp = actual.copy()
    for i in range(len(actual)):
        if(actual[i]==value):
            temp[i]=1
        else:
            temp[i]=0
    return temp


# In[228]:


testDT0=convert(predicted_test_DT,0)
testDT1=convert(predicted_test_DT,1)
testDT2=convert(predicted_test_DT,2)


# In[229]:


FalsePositiveDT0,TruePositiveDT0,threshold=roc_curve(test_labels_dummies[:,[0]],testDT0)
py.plot(FalsePositiveDT0,TruePositiveDT0,label="Class 0")
FalsePositiveDT1,TruePositiveDT1,threshold=roc_curve(test_labels_dummies[:,[1]],testDT1)
py.plot(FalsePositiveDT1,TruePositiveDT1,label="Class 1")
FalsePositiveDT2,TruePositiveDT2,threshold=roc_curve(test_labels_dummies[:,[2]],testDT2)
py.plot(FalsePositiveDT2,TruePositiveDT2,label="Class 2")
py.title("ROC CURVE-DT")
py.xlabel("False Positive Rate")
py.ylabel("True Positive Rate")
py.legend()
py.show()


# In[ ]:




