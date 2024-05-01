# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:53:18 2024
@author: Vaibhav Bhorkade
"""

# Voting
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
iris=datasets.load_iris()
X,y=iris.data[:,1:3],iris.target # taking entire data as training data
clf1=LogisticRegression()
clf2=RandomForestClassifier(random_state=1)
clf3=GaussianNB()

#############################################

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