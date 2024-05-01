# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 00:27:24 2024

@author: Vaibhav Bhorkade
"""

"""
A cloth manufacturing company is interested to know about
the different attributes contributing to high sales. 
Build a decision tree & random forest model with Sales 
as target variable (first convert it into categorical variable).
"""
"""
Business Objective 
Minimize : Minimize costs of cloths.
Maximaze : Maximize overall Sales.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("Company_Data.csv")

df.head(10)
df.tail()

# 5 number summary
df.describe()

df.shape
# 400 rows and 11 columns
df.columns
'''
['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
'''

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 no null values

import seaborn as sns
import matplotlib.pyplot as plt
# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# boxplot
# boxplot on Sales column
sns.boxplot(df.Sales)
# In Sales column 2 outliers 

sns.boxplot(df.Income)
# In Income column no outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['Income'],kde=True)
# normally right skew and the distributed

sns.histplot(df['Sales'],kde=True)
# skew and the distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
df.dtypes
# Some columns in int, float data types and some Object

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
lb=LabelEncoder()
df["SheveLoc"]=lb.fit_transform(df["ShelveLoc"])
df["US"] = lb.fit_transform(df["US"])
df["Urban"]= lb.fit_transform(df["Urban"])


X=df.iloc[:,:11].values  
y=df.SheveLoc

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


