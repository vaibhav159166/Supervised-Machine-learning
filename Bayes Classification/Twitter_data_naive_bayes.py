# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:50:07 2024

@author: Vaibhav Bhorkade
"""
"""
Problem Statement: -
In this case study, you have been given Twitter data collected 
from an anonymous twitter handle. With the help of a Naïve Bayes 
model, predict if a given tweet about a real disaster is real or fake.
1 = real tweet and 0 = fake tweet

"""
"""
Business Objective : 
Predict whether a disaster is real or fake
Minimize : Minimize misclassification of the disaster.
Maximaze : Maximize overall accuracy and Increase the efficiency of model to predict
           disaster is real or fake.
"""
"""
Data Dictionary

Features      Typ
id          continuous    
keyword     Qualitative
location    Qualitative
text        Qualitative
target      Categorical
"""

# EDA - Exploratory data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Disaster_tweets_NB.csv")
# head and tail data
df.head(10)
df.tail()

# 5 number summary
df.describe()
# min is 0

df.shape
# 7613 rows and 5 columns
df.columns
'''
['id', 'keyword', 'location', 'text', 'target']
'''

# check for null values
df.isnull()
# False
df.isnull().sum()
# locations 2533 null values
df.dropna(inplace=True)
# Now all null value droped

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# boxplot
# boxplot on df 
sns.boxplot(df)
# There is no outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['target'],kde=True)
# skew and the distributed
# May be symmetric

sns.histplot(df['id'],kde=True)
# not skew and the distributed

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
# output is 0

#Following are the columns of object type
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import re
##Since there are no numeric data hence further EDA is not possible
#let us conver the text messages to TFIDF 
#Let us clean the data
def cleaning_text(i):
    i=re.sub("[^A-Za-z""]+"," ",i).lower()
    #Let us declare empty list
    w=[]
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return("  ".join(w))
#Let us check the function
cleaning_text("This is just trial text to check the cleaning_text method")
cleaning_text("Hi how are you ,I am sad")
#There could messages which will create empty spaces after the cleaning
#Now first let us apply to the tweet.text column
df.text=df.text.apply(cleaning_text)
#The numer rows of tweet is 7613
#Let us first drop the columns which are not useful
df.drop(["id","keyword","location"],axis=1,inplace=True)
#There could messages which will create empty spaces after the cleaning
tweet=df.loc[df.text !="",:]
#The number of rows are reduced to 7610 after this cleaning
###########################################
###let us first split the data in training set and Testing set
from sklearn.model_selection import train_test_split
tweet_train,tweet_test=train_test_split(tweet,test_size=0.2)
#####################################
#let us first tokenize the message
def split_into_words(i):
    return[word for word in i.split(" ")]
#This is tokenization or custom function will be used for CountVectorizer
tweet_bow=CountVectorizer(analyzer=split_into_words).fit(tweet.text)
#This is model which will be used for creating count vectors
#let us first apply to whole data
all_tweet_matrix=tweet_bow.transform(tweet.text)
#Now let us apply to training messages
train_tweet_matrix=tweet_bow.transform(tweet_train.text)
#similarly ,let us apply to test_tweet
test_tweet_matrix=tweet_bow.transform(tweet_test.text)
##Let us now apply to TFIDF Transformer
tfidf_transformer=TfidfTransformer().fit(all_tweet_matrix)
##This is being used as model.let us apply to train_tweet_matrix
train_tfidf=tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape
#let us now apply it to test_tweet_matrix
test_tfidf=tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape
##################################################
#let us now apply it to Naive model
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
#let us train the model
classifier_mb.fit(train_tfidf,tweet_train.target)
#############################
# let us now evaluate the model with test data
test_pred_m=classifier_mb.predict(test_tfidf)
##Accuracy of the prediction
accuracy_test_m=np.mean(test_pred_m==tweet_test.target)
accuracy_test_m
#To find the confusion matrix
from sklearn.metrics import accuracy_score
pd.crosstab(test_pred_m,tweet_test.target)
###let us check the wrong classified actual is fake and predicted is not fake is 64
#actual is not fake but predicted is fake is 255 ,this is not accepted
######################################
#Let us evaluate the model with train data
train_pred_m=classifier_mb.predict(train_tfidf)
accuracy_train_m=np.mean(train_pred_m==tweet_train.target)
accuracy_train_m
###let us check the confusion matrix
pd.crosstab(train_pred_m,tweet_train.target)
classifier_mb_lap=MB(alpha=0.25)
classifier_mb_lap.fit(train_tfidf,tweet_train.target)
###Evaluation on test data
test_pred_lap=classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap=np.mean(test_pred_lap==tweet_test.target)
accuracy_test_lap
from sklearn.metrics import accuracy_score
pd.crosstab(test_pred_lap,tweet_test.target)
###let us check the wrong classified actual is fake and predicted is not fake is 103
#actual is not fake but predicted is fake is 215 ,this is not accepted
######################
#Training data accuracy
train_pred_lap=classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap=np.mean(train_pred_lap==tweet_train.target)
accuracy_train_lap
pd.crosstab(train_pred_lap,tweet_train.target)