# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:15:09 2024

@author: Vaibhav Bhorkade
"""
"""
Prepare a classification model using the Naive Bayes 
algorithm for the salary dataset. Train and test datasets 
are given separately. Use both for model building.  
"""
"""
Business Objective : 
Predict whether an individual's salary is greater than 
50K or not based on the given features.
 
Minimize : Minimize misclassification of salary.
Maximaze : Maximize overall accuracy and Increase the efficiency of model to predict
           a salary.

Business constraints:
Contraints is the limited computational resources.
"""
"""
Data Dictionary
        Features      Type          Relevance
0             age     Quantative    Relevant
1       workclass  Qualititative    Relevant
2       education        Ordinal    Relevant
3     educationno       Discrete    Relevant
4   maritalstatus        nominal  Irrelevant
5      occupation  Qualititative  Irrelevant
6    relationship  Qualititative  Irrelevant
7            race     Qualitatve  Irrelevant
8             sex        Nominal    Relevant
9     capitalgain       Discrete    Relevant
10    capitalloss       discrete    Relevant
11   hoursperweek      continous    Relevant
12         native    Qualitative  Irrelevant
13         Salary       Discrete    Relevant


"""

# EDA - Exploratory data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df_train=pd.read_csv("SalaryData_Train.csv")
df_test=pd.read_csv("SalaryData_Test.csv")
df_train.head(10)
df_train.tail()

df_test.head(10)

# 5 number summary
df_train.describe()
df_test.describe()
df_train.shape
# 30161 rows and 14 columns
df_test.shape
# 15060 rows and 14 columns
df_train.columns
'''
'age', 'workclass', 'education', 'educationno', 'maritalstatus',
 'occupation', 'relationship', 'race', 'sex', 'capitalgain',
'capitalloss', 'hoursperweek', 'native', 'Salary'''

df_test.columns
# Same columns

# check for null values
df_train.isnull()
# False
df_train.isnull().sum()
# 0 , no null values
df_test.isnull().sum()
# 0 no null values
# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df_train);
plt.show()

# boxplot
# boxplot on age column
sns.boxplot(df_train.age)
# In age column many outliers 

sns.boxplot(df_test.age)
# In age column many outliers

# boxplot on df column
sns.boxplot(df_train)
# There is outliers on all columns

# boxplot on educationno column
sns.boxplot(df_train.educationno)
# There is 2 outliers on column
sns.boxplot(df_test.educationno)
# There is 2 outliers on column

# histplot - show distributions of datasets
sns.histplot(df_train['educationno'],kde=True)
# skew and the distributed
# May be symmetric

sns.histplot(df_test['age'],kde=True)
# Right skew and the distributed

sns.histplot(df_train,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
df_train.dtypes
df_test.dtypes
# Some columns in int data types and some Object

# Identify the duplicates
duplicate=df_test.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
# output is 930
# delete duplicate values
salary_test=df_test.drop_duplicates()
duplicate=salary_test.duplicated()
sum(duplicate)
# sum is 0.

duplicate=df_train.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
# output is 930
# delete duplicate values
salary_train=df_train.drop_duplicates()
duplicate=salary_train.duplicated()
sum(duplicate)

# sum is 0.
#Following are the columns of object type
#Let us apply label encoder to input features
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
#This is model of label_encoder which is applied to all the object type columns
for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])

#Now let us apply normalization function
def norm_funct(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
salary_train_norm=norm_funct(salary_train)       
salary_test_norm=norm_funct(salary_test)        
##################################################
#Now let us designate train data and Test data
salary_train_col=list(salary_train.columns)
train_X=salary_train[salary_train_col[0:13]]
train_y=salary_train[salary_train_col[13]]

salary_test_col=list(salary_test.columns)
test_X=salary_test[salary_test_col[0:13]]
test_y=salary_test[salary_test_col[13]]
############################################
##	Model Building
#Build the model on the scaled data (try multiple options).
#Build a Naïve Bayes model.
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()

classifier_mb.fit(train_X,train_y)
#Let us now evaluate on test data
test_pred_m=classifier_mb.predict(test_X)
##Accuracy of the prediction
accuracy_test_m=np.mean(test_pred_m==test_y)
accuracy_test_m
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m,test_y)
pd.crosstab(test_pred_m,test_y)
###let us check the wrong classified actual is grater than 50 and predicted is less than 50 is 469
#actual salary prediction less than 50 but predicted is 'greater than 50' is 2920 ,this is not accepted
################################################
#Let us now evaluate on train data
train_pred_m=classifier_mb.predict(train_X)
##Accuracy of the prediction
accuracy_train_m=np.mean(train_pred_m==train_y)
accuracy_train_m
#0.7729
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(train_pred_m,train_y)
pd.crosstab(train_pred_m,train_y)
###let us check the wrong classified actual is 'grater than 50' and predicted is 'less than 50' is 936
#actual salary prediction less than 50 but predicted is 'greater than 50' is 5913 ,this is not accepted
################################################
##Multinomial Naive Bayes with laplace smoothing
###in order to address problem of zero probability laplace smoothing is used
classifier_mb_lap=MB(alpha=0.75)
classifier_mb_lap.fit(train_X,train_y)

#Let us now evaluate on test data
test_pred_lap=classifier_mb_lap.predict(test_X)
##Accuracy of the prediction
accuracy_test_lap=np.mean(test_pred_lap==test_y)
accuracy_test_lap
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap,test_y)
pd.crosstab(test_pred_lap,test_y)
###let us check the wrong classified actual is 'grater than 50' and predicted is 'less than 50 is 469
#actual salary prediction less than 50 but predicted is 'greater than 50' is 2920 ,this is not accepted
############################################
#Let us now evaluate on train data
train_pred_lap=classifier_mb.predict(train_X)
##Accuracy of the prediction
accuracy_train_lap=np.mean(train_pred_m==train_y)
accuracy_train_m
#0.7729
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(train_pred_lap,train_y)
pd.crosstab(train_pred_lap,train_y)

