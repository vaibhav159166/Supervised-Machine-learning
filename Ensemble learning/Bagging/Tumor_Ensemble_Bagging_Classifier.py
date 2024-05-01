# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:17:48 2024
@author: Vaibhav Bhorkade
"""
"""
Most cancers form a lump called a tumour. But not all lumps are cancerous. 
Doctors extract a sample from the lump and examine it to find out if it’s 
cancer or not. Lumps that are not cancerous are called benign (be-NINE). 
Lumps that are cancerous are called malignant (muh-LIG-nunt). 
Obtaining incorrect results (false positives and false negatives) 
especially in a medical condition such as cancer is dangerous. So, 
perform Bagging, Boosting, Stacking, and Voting algorithms to increase 
model performance and provide your insights in the documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Tumor_Ensemble.csv")
df.head()
df.shape
# 569 rows and 32 columns

# Check for null values
df.isnull().sum()
# No null values

df.describe()
df.columns


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df['diagnosis']=labelencoder.fit_transform(df['diagnosis'])

df.diagnosis.value_counts()
# 0    357
# 1   212

# There is slight imbalance in our dataset but since it is 

# Train test split
X=df.drop("diagnosis",axis="columns")
y=df.diagnosis


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
# In order to make your data balanced while splitting, you can use 
# stratify
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X_scaled, y, stratify=y,random_state=10)
X_train.shape
X_test.shape
y_train.value_counts()

y_test.value_counts()

# Train using stand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
# Here k fold cross validation is used
scores=cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
scores
scores.mean()
# Accuracy = 0.93

# Train using Bagging
from sklearn.ensemble import BaggingClassifier

bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)

bag_model.fit(X_train, y_train)
bag_model.oob_score_
# 0.94

# OOB 
bag_model.score(X_test, y_test)
# 0.95
#cross validation
bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
scores=cross_val_score(bag_model, X,y,cv=5)
scores
scores.mean()
# 0.95
# Train using Random forest
from sklearn.ensemble import RandomForestClassifier

scores=cross_val_score(RandomForestClassifier(n_estimators=50), X,y,cv=5)
scores.mean()

# Boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Create adaboost classifier
ada_model=AdaBoostClassifier(n_estimators=100,learning_rate=1)
# n_estimators=number of weak learners
# learning rate, it contributes weights of weak learners, bydefault 
# Train the model
model=ada_model.fit(X_train,y_train)
# predict the results
y_pred=model.predict(X_test)
print("Accuracy",metrics.accuracy_score(y_test, y_pred))
# Let us try for another base model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
# here base model is changed
Ada_model=AdaBoostClassifier(n_estimators=50,base_estimator=lr,learning_rate=1)
model=ada_model.fit(X_train,y_train)
