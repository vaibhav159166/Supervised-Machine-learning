# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 08:15:13 2024
@author: Vaibhav Bhorkade
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
claimants=pd.read_csv("claimants.csv")
#There are CLMAGE and LOSS are having continuous data rest are
#verify the dataset,where CASENUM is  not really useful so drop
c1=claimants.drop('CASENUM',axis=1)
c1.head(11)
c1.describe()
#let us check whether there are null values
c1.isna().sum()
#there are several null values
#If we will use dropna() function we will lose 290 data points
#hence we will go for imputation
c1.dtypes
mean_value=c1.CLMAGE.mean()
mean_value
#Now let us impute the same
c1.CLMAGE=c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()
#hence all null values of CLMAGE has been filled by mean value
#for columns where there are discrete values, we will apply mode
mode_CLMSEX=c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX=c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()
#CLMINSUR is also categorical data hence mode imputation is applied
mode_CLMINSUR=c1.CLMINSUR.mode()
mode_CLMINSUR
c1.CLMINSUR=c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()
#SEATBELT is a categorical data hence go for mode imputation
mode_SEATBELT=c1.SEATBELT.mode()
mode_SEATBELT
c1.SEATBELT=c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()
#Now the person who met an accident will hire the attorney or no
#Let us build the model
logit_model=sm.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=c1).fit()
logit_model.summary()
#In logistic regression we do not have R square values,only check p=values
#Statistically insignificant ignore and proceed
logit_model.summary2()
# Here we are going to check AIC value, which stands for AKaike Information Criterion
#is a mathematical method for evaluation how well the model fits the data
#A lower the score more the better the model,AIC scores are only useful
#with other AIC scores for the same dataset

#Now let us go for predictions
pred=logit_model.predict(c1.iloc[:,1:])
#here we are applying all rows columns from1 as column 0 is ATTORNEY
#target value

#let us check the performance of the model
fpr,tpr,thresholds=roc_curve(c1.ATTORNEY,pred)

#we arw applying actual values and predicted values so as to get
#false positive rate,true poositive rate and threshold
#The optimal cutoff value is the point where there is high true positive
#you can use the below code to get the values:
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold

#ROC:Receiver Operating Characteristics curve in logistic regression are
#determining the best cutoff/threshold value
import pylab as pl
i=np.arange(len(tpr))#index for df

#here tpr is of 559, so it will create a scale from 0 to 558
roc=pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                  'tpr' : pd.Series(tpr,index=i),
                  '1-fpr':pd.Series(1-fpr,index=i),
                  'tf' :pd.Series(tpr-(1-fpr), index=i),
                  'threshold':pd.Series(thresholds,index=i)})

#we want to create a dataframe which comprises of columns  fpr,
#tpr,1-fpr,tpr-(1-fpr=tf)
#the optimal cut off would be where tpr is high anf fpr is low
#tpr-(1-fpr) is zero or near to zero is the optimal cut off point

#plot ROC curve
plt.plot(fpr,tpr)
plt.xlabel('False positive rate');plt.ylabel("True Positive rate")
roc.iloc[(roc.tf-0)].abs()
plt.plot(fpr,tpr)
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc=auc(fpr,tpr)
print('Area under the curve:%f'%roc_auc)

#plot tpr vs (1-fpr)
fig,ax=pl.subplots()
pl.plot(roc['tpr'],color='red')
pl.plot(roc['1-fpr'],color='blue')
pl.xlabel('1-False Positive rate')
pl.ylabel('True Positive rate')
plt.title('Reciever Operating Characteristics')
ax.set_xticklabels([])
#the optimal cutoff point is one where tpr is high and fpr is low
#the optimal cutoff point is 0.31762
#so anything above this can be labeled as 0 else 1
#you can see from output that where TPR is crossong 1-FPR
#FPR is 36% and TPR is nearest to zero



#filling all the cells with  zeros
c1['pred']=np.zeros(1340)
c1.loc[pred>optimal_threshold,'pred']=1

#Now let us check the classification report
classification=classification_report(c1["pred"],c1["ATTORNEY"])
classification

#splitting the data into train and test
#splitting the data into train and test
train_data,test_data=train_test_split(c1,test_size=0.3)

#model building
model=sm.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=train_data).fit()
model.summary()
#p=values arethe below the condition of 0.05
#but SEATBELT has got statistically insignificant
model.summary2()
#AIC value is 1110.3782,AICscore are useful in comparison with other
#lower the AIC score better the model
#let us go for predictions
test_pred=logit_model.predict(test_data)
#creating new columnfor storing pedictedclass of ATTORNEY
test_data['test_pred']=np.zeros(402)
test_data.loc[test_pred>optimal_threshold,"test_pred"]=1

#Confusion Matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix
accuracy_test=(143+151)/(402)#Add current Values
accuracy_test

#Classification report
classification_test=classification_report(test_data["test_pred"],test_data["ATTORNEY"])
classification_test

#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data["ATTORNEY"],test_pred)

#plot ROC Curve
plt.plot(fpr,tpr);plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate")

#AUC
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test

#prediction on train data
train_pred=logit_model.predict(train_data)
#Creating new column for storing predicted class of ATTORNEY
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.ATTORNEY)
confusion_matrix
accuracy_train=(315+347)/(938)
accuracy_train
#0.072174, this is going to  change with everytime when you 


#####################################################################
#Classifcation report
classification_train=classification_report(train_data["train_pred"],
                                           train_data["ATTORNEY"])
classification_train
#Accuracy=0.69

#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(train_data["ATTORNEY"],train_pred)

#plotROC Curve
plt.plot(fpr,tpr);plt.xlabel("False Positive Rate");plt.ylabel("True Psitive Rate")

#AUC
roc_auc_train=metrics.auc(fpr,tpr)
roc_auc_train

















