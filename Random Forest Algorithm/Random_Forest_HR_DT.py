# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 00:21:40 2024

@author: Vaibhav Bhorkade
"""
"""
In the recruitment domain, HR faces the challenge of predicting 
if the candidate is faking their salary or not. For example, a 
candidate claims to have 5 years of experience and earns 70,000 per 
month working as a regional manager. The candidate expects more money 
than his previous CTC. We need a way to verify their claims
(is 70,000 a month working as a regional manager with an experience 
of 5 years a genuine claim or does he/she make less than that?) 
Build a Decision Tree and Random Forest model with monthly 
income as the target variable. 
"""
"""
Business Objective
Minimize: To reduce or keep something as small as possible, 
often referring to costs, risks, or inefficiencies in a business process.

Maximize: To increase or optimize something to the greatest extent, 
often used in the context of profits, efficiency, or positive outcomes.
"""
"""
Data Dictionary

 Features                                      Type             Relevance
0   Position of the employee                 Qualititative data  Relevant
1   no of Years of Experience of employee    Continious data     Relevant
2   monthly income of employee               Continious data     Relevant

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv("HR_DT.csv")

df.head(10)
df.tail()

# 5 number summary
df.describe()

df.shape
# 196 rows and 3 columns

df.columns
'''
['Position of the employee', 'no of Years of Experience of employee',
' monthly income of employee']
'''

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 no null values

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# Column name 
df.columns = [
    'Position_of_the_employee',
    'Experience_of_employee',
    'monthly_income',
]

# Now you can access the columns
print(df.columns)

# boxplot
# boxplot on Experience_of_employee column
sns.boxplot(df.Experience_of_employee)
# In Experience_of_employee column no outliers 

sns.boxplot(df.monthly_income)
# In monthly_income column no outliers

# boxplot on df column
sns.boxplot(df)
# There is no outliers columns

# histplot - show distributions of datasets
sns.histplot(df['monthly_income'],kde=True)
# right skew and the distributed

sns.histplot(df['Experience_of_employee'],kde=True)
# right skew and the distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
df.dtypes
# Some columns in int data types and some Object

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
# sum is 0.

df.isnull().sum()
df.dropna()
df.columns

# Converting into binary
# Convert categorical column to numerical using LabelEncoder
le = LabelEncoder()
df["Position_of_the_employee"] = le.fit_transform(df["Position_of_the_employee"])

X=df.iloc[:,:3].values
#  X=df.iloc[:,:2].values
y=df.monthly_income

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
#n_estimator:number of trees in the forest
model.fit(X_train,y_train)

model.score(X_test, y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)
cm
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")


