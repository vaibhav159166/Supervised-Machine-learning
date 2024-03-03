# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:54:55 2024

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
Minimize: To reduce costs, risks, or inefficiencies in a business process.
Maximize: To increase profits, efficiency, or positive outcomes.
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

# Split the data into predictors and target
colnames = list(df.columns)
predictors = colnames[:2]
target = colnames[2]

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.3)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(train[predictors], train[target])

# Make predictions on the test set
preds_test = model.predict(test[predictors])

# Evaluate the model
accuracy_test = accuracy_score(test[target], preds_test)
conf_matrix_test = confusion_matrix(test[target], preds_test)

print("Accuracy on Test Set:", accuracy_test)
print("Confusion Matrix on Test Set:\n", conf_matrix_test)
