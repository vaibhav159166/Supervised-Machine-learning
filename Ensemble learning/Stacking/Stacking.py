# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:17:32 2024
@author: Vaibhav Bhorkade
"""
# Stacking
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
# conda install mlxtend
from mlxtend.classifier import StackingClassifier
import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
iris=datasets.load_iris()
X_train,y_train=iris.data[:,1:3],iris.target
weak_11=KNeighborsClassifier(n_neighbors=1)
weak_12=RandomForestClassifier(random_state=1)
weak_13=GaussianNB()
#########################################################
meta_1=LogisticRegression()
stackingclf=StackingClassifier(classifiers=[weak_11,weak_12,weak_13], meta_classifier=meta_1)
##########################################################
print("After three fold cross validation")
for iterclf,iterlabel in zip([weak_11,weak_12,weak_13,stackingclf],['K-nearest Neighbor Model','Random Forest Model','Naive Bayes Model','Stacking classifier model']):
    scores=model_selection.cross_val_score(iterclf,X_train,y_train, cv=3,scoring='accuracy')
    print("Accuracy : ",scores.mean(),"for ",iterlabel)