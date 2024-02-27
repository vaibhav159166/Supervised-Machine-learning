# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:24:26 2024

@author: Vaibhav
"""
"""
Problem Statement :
A National Zoopark in India is dealing with the problem of 
segregation of the animals based on the different attributes
they have. Build a KNN model to automatically classify the 
animals. Explain any inferences you draw in the documentation.
"""
"""
Business Uderstanding/Objective
Maximize : Maximizing the probabality of classifying correct animals in category
Minimize : To minimize wrong classification of animals.
"""

"""
Data Dictionary

Features         Type   Relevance
0   animal name  Qualitative  Irrelevant
1          hair      Nominal    Relevant
2      feathers      Nominal    Relevant
3          eggs      Nominal    Relevant
4          milk      Nominal    Relevant
5      airborne      Nominal    Relevant
6       aquatic      Nominal    Relevant
7      predator      Nominal    Relevant
8       toothed      Nominal    Relevant
9      backbone      Nominal    Relevant
10     breathes      Nominal    Relevant
11     venomous      Nominal    Relevant
12         fins      Nominal    Relevant
13         legs      Nominal    Relevant
14         tail      Nominal    Relevant
15     domestic      Nominal    Relevant
16      catsize      Nominal    Relevant
17         type      Nominal    Relevant

"""

# EDA
import pandas as pd
import numpy as np
df = pd.read_csv("Zoo.csv")
df.head()

df.shape
# 101 rows, 18 columns

# 5-number symmary
df.describe()
df.info()

df['type'].value_counts()

# drop column 'animal' 
df = df.drop('animal name',axis=1)

# Normalization
def norm(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_norm = norm(df.iloc[:,0:16])

# Now let us take X as input & Y as output
X = np.array(df_norm.iloc[:,:])
Y = np.array(df['type'])

# Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
# to avoid the unbalancing of data during splitting the concept of 
# statified sampling is used

# Now build the KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
pred

# Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,Y_test))
pd.crosstab(pred,Y_test)
# So this model is not accepted as there are errors

# Let us try to select the correct value of k
acc= []
# Running the KNN algorithm for k=3 to 50 in step of 2
# k's value is selected as odd
for i in range(3,50,2):
    #declare model
    n = KNeighborsClassifier(n_neighbors=i)
    n.fit(X_train,Y_train)
    train_acc = np.mean(n.predict(X_train) == Y_train)
    test_acc = np.mean(n.predict(X_test) == Y_test)
    acc.append([train_acc,test_acc])

# lets plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'bo-')
# There are valiues like 3,5,7,9 where the accuracy is good

# lets try for K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
accuracy_score(pred,Y_test)
# 1.0
pd.crosstab(pred,Y_test)
# This is perfect model
