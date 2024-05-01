# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:00:06 2024
@author: Vaibhav Bhorkade
"""

"""
Most cancers form a lump called a tumour. But not all lumps are 
xcancerous. Doctors extract a sample from the lump and examine it 
to find out if it’s cancer or not. Lumps that are not cancerous are 
called benign (be-NINE). Lumps that are cancerous are called malignant 
(muh-LIG-nunt). Obtaining incorrect results (false positives and false 
negatives) especially in a medical condition such as cancer is dangerous. 
So, perform Voting algorithms to increase model performance and provide 
your insights in the documentation.
"""

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
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

