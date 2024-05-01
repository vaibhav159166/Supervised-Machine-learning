# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 00:14:04 2024

@author: Vaibhav Bhorkade
"""

"""
Build a Decision Tree & Random Forest model on the fraud data. 
Treat those who have taxable_income <= 30000 as Risky and others 
as Good (discretize the taxable income column).
"""
"""
Business Objective: Attaining a defined goal for the organization.

Minimize: Decreasing or eliminating certain factors or costs.
Maximize: Enhancing or optimizing specific aspects for the greatest benefit.
"""

"""
Data Dictionary

 Features               Type          Relevance
0        Undergrad       Nominal data  Relevant
1   Marital.Status   Categorical data  Relevant
2   Taxable.Income  Quantititave data  Relevant
3  City.Population  Quantitative data  Relevant
4  Work.Experience  Quantitative data  Relevant
5            Urban       Nominal data  Relevant
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("Fraud_check.csv")

df.head(10)
df.tail()

# 5 number summary
df.describe()

df.shape
# 600 rows and 6 columns

df.columns
'''
['Undergrad', 'Marital.Status', 'Taxable.Income', 'City.Population',
'Work.Experience', 'Urban']
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


df.columns = [
    'Undergrad',
    'Marital_Status',
    'Income',
    'Population',
    'Experience',
    'Urban'
]

# Now you can access the columns
print(df.columns)

# boxplot
# boxplot on Income column
sns.boxplot(df.Income)
# In Income column 1 outliers 

sns.boxplot(df.Population)
# In Population column no outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['Income'],kde=True)
# right skew and the distributed

sns.histplot(df['Population'],kde=True)
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
# Normalize data 
# Normalize the data using norm function
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Apply the norm_fun to data 
df1=norm_fun(df.iloc[:,2:5])

df['Undergrad']
df1['Undergrad']=df['Undergrad']

df['Marital_Status']
df1["Marital_Status"]=df["Marital_Status"]

df["Urban"]
df1["Urban"]=df["Urban"]



df.isnull().sum()
df.dropna()
df.columns

# Converting into binary
lb=LabelEncoder()
df1["Undergrad"]=lb.fit_transform(df1["Undergrad"])
df1["Marital_Status"]=lb.fit_transform(df1["Marital_Status"])
df1["Urban"]=lb.fit_transform(df1["Urban"])

X=df1.iloc[:,:5].values  
y=df1.Urban

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


