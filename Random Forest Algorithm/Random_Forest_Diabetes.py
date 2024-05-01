# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:41:42 2024

@author: Vaibhav Bhorkade
"""
"""
Divide the diabetes data into train and test datasets and 
build a Random Forest and Decision Tree model with Outcome 
as the output variable. 
"""
"""
Business Objective: It could be related to improving efficiency,
increasing revenue, reducing costs, or enhancing customer 
satisfaction.

Minimize: Minimizing the number of false negatives.
Maximize: Maximizing the true positive rate.
"""
"""
Data Dictionary

        Features                 Type Relevance
0       Number of times pregnant   Quantitative data  Relevant
1   Plasma glucose concentration   Quantitative data  Relevant
2       Diastolic blood pressure   Quantitative data  Relevant
3    Triceps skin fold thickness   Quantitative data  Relevant
4           2-Hour serum insulin   Quantitative data  Relevant
5                Body mass index   Quantitative data  Relevant
6     Diabetes pedigree function   Quantitative data  Relevant
7                    Age (years)   Quantitative data  Relevant
8                 Class variable         Nominal data  Relevant

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("Diabetes.csv")

df.head(10)
df.tail()

# 5 number summary
df.describe()

df.shape
# 768 rows and 9 columns
df.columns
'''
[' Number of times pregnant', ' Plasma glucose concentration',
       ' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin', ' Body mass index',
       ' Diabetes pedigree function', ' Age (years)', ' Class variable']
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

# Replace column names
# df.columns = df.columns.str.replace(' ', '_')
# df.columns

df.columns = [
    'Pregnant',
    'Glucose',
    'BloodPressure',
    'thickness',
    'insulin',
    'BMS',
    'Diabetes',
    'age',
    'class_variable'
]

# Now you can access the columns
print(df.columns)

# boxplot
# boxplot on Pregnant column
sns.boxplot(df.Pregnant)
# In Pregnant column 3 outliers 

sns.boxplot(df.age)
# In Income column many outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['age'],kde=True)
# normally right skew and the distributed

sns.histplot(df['Pregnant'],kde=True)
# right skew and the distributed

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
# Normalize data 
# Normalize the data using norm function
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Apply the norm_fun to data 
df1=norm_fun(df.iloc[:,:8])

df1.isnull().sum()
df1.dropna()
df1.columns

X=df1.iloc[:,:7].values  
y=df1.class_variable

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