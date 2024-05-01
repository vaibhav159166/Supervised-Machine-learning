# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:00:05 2024
@author: Vaibhav Bhorkade
"""

"""
Given is the diabetes dataset. Build an ensemble model to correctly 
classify the outcome variable and improve your model prediction by 
using GridSearchCV. You must Voting on the dataset.
"""

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

df=pd.read_csv("Diabeted_Ensemble.csv")
df.head()
df.shape
# 768 rows and 9 columns

# Check for null values
df.isnull().sum()
# No null values

df.describe()
df.columns

# Rename column names 
df.columns=[
    'N_pregnant', 'glucose',
           'blood_pressure', 
           'skin_fold_thickness',
           'insulin', 'BMI',
           'Diabetes_function', 
           'Age', 
           'Class_variable'
           ]
df.columns

df.Class_variable.value_counts()
# 0    500
# 1    268

# Extract features and target variable
X = df.drop('Class_variable', axis=1)
y = df['Class_variable']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

print("After five fold cross validation")
labels=['Logistic Regression','Random Forest model','Naive Bayes Model']
for clf,label in zip([clf1,clf2,clf3],labels):
    scores=model_selection.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print("Accuracy : ",scores.mean(),"for ",label)

voting_clf_hard=VotingClassifier(estimators=[(labels[0],clf1),(labels[1],clf2),(labels[2],clf3)],voting="hard")

voting_clf_soft=VotingClassifier(estimators=[(labels[0],clf1),(labels[1],clf2),(labels[2],clf3)],voting="soft")

labels_new=['Logistic Regression','Random Forest model','Naive Bayes model','voting clf hard','voting clf soft']
for clf,label in zip([clf1,clf2,clf3,voting_clf_hard,voting_clf_soft],labels_new):
    scores=model_selection.cross_val_score(clf,X,y,cv=5, scoring='accuracy')
    print("Accuracy : ",scores.mean(),'for ',label)  