# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:53:46 2024

@author: Vaibhav Bhorkade
"""
"""
problem ststement :
1.	A glass manufacturing plant uses different earth elements to 
design new glass materials based on customer requirements. 
For that, they would like to automate the process of classification
as it’s a tedious job to manually classify them. Help the company 
achieve its objective by correctly classifying the glass type based 
on the other features using KNN algorithm.
"""
"""
Business Uderstanding/Objective
Maximize : Maximizing the probabality of classifying the correct earth elements and
on the basis of features give correct output.
Minimize : To minimize wrong classification of elements.
"""

"""
Data Dictionary

Feature        Type   Relevance
0      RI  Continious  Relevant
1      Na  Continious  Relevant
2      Mg  Continious  Relevant
3      Al  Continious  Relevant
4      Si  Continious  Relevant
5       K  Continious  Relevant
6      Ca  Continious  Relevant
7      Ba  Continious  Relevant
8      Fe  Continious  Relevant
9    Type     Nominal  Relevant

"""

# EDA 
import pandas as pd
import numpy as np 
glass=pd.read_csv("glass.csv")
glass

glass.describe()
glass.info()

glass['Type'].value_counts()

#normalization
def norm(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#let us apply normlization function to the dataset
glass_norm = norm(glass.iloc[:,0:9])

#now let us take X as input & Y as output
X = np.array(glass_norm.iloc[:,:])
Y = np.array(glass['Type'])

#split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
#to avoid the unbalancing of data during splitting the concept of 
#statified sampling is used

#now build the KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
pred

#now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,Y_test))
pd.crosstab(pred,Y_test)


#let us try to select the correct value of k
acc = []
#running the KNN algorithm for k=3 to 50 in step of 2
#k's value is selected as odd
for i in range(3,50,2):
    #declare model
    n = KNeighborsClassifier(n_neighbors=i)
    n.fit(X_train,Y_train)
    train_acc = np.mean(n.predict(X_train) == Y_train)
    test_acc = np.mean(n.predict(X_test) == Y_test)
    acc.append([train_acc,test_acc])

#lets plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'bo-')
#there are valiues like 3,5,7,9 where the accuracy is good

#lets try for K=3
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
accuracy_score(pred,Y_test)
#0.5581395348837209
pd.crosstab(pred,Y_test)





























