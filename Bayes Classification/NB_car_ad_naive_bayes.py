# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 8:15:09 2024

@author: Vaibhav Bhorkade
"""
"""
Problem Statement: -
This dataset contains information of users in a social network. 
social network has several business clients which can post ads 
on it. One of the clients has a car company which has just 
launched a luxury SUV for a ridiculous price. Build a Bernoulli 
Naïve Bayes model using this dataset and classify which of the users 
of the social network are going to purchase this luxury SUV. 1 implies
that there was a purchase and 0 implies there wasn’t a purchase.
  
"""
"""
Business Objective : 
Predict whether a customer will make a purchase (Purchased = 1) 
based on their gender, age, and estimated salary.
 
Minimize : Minimize misclassification on the base of salry.
Maximaze : Maximize overall accuracy and Increase the efficiency of model to predict
           a an=ble to purchase or not.
"""
"""
Data Dictionary

          Features          Type   Relevance
0          User ID     Numerical  Irrelevant
1           Gender       Nominal    Relevant
2              Age    Quantative    Relevant
3  EstimatedSalary    Quantative    Relevant
4        Purchased       Nominal    Relevant

"""

# EDA - Exploratory data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("NB_Car_Ad.csv")
df.head(10)
df.tail()

# 5 number summary
df.describe()
# min age-18 , EstimatedSalary-15000

df.shape
# 400 rows and 5 columns
df.columns
'''
['User ID', 'Gender', 'Age', 'EstimatedSalary', 
'Purchased']
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

# boxplot
# boxplot on age column
sns.boxplot(df.Age)
# In age column no outliers 

sns.boxplot(df.EstimatedSalary)
# In age column no outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['Age'],kde=True)
# normally right skew and the distributed
# May be symmetric

sns.histplot(df['EstimatedSalary'],kde=True)
# skew and the distributed

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
# output is 930
# delete duplicate values
df1=df.drop_duplicates()
duplicate=df.duplicated()
sum(duplicate)
# sum is 0.
#Following are the columns of object type
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Selecting relevant features (User ID, Gender, Age, EstimatedSalary) and target variable (Purchased)
features = df[['Gender', 'Age', 'EstimatedSalary']]
target = df['Purchased']

# Splitting the data into training and testing sets
social_train, social_test = train_test_split(df, test_size=0.2, random_state=42)

# Creating matrix of token counts for the entire feature set
def split_into_words(text):
    return [word for word in str(text).split()]

social_bow = CountVectorizer(analyzer=split_into_words).fit(features.apply(lambda x: ' '.join(x.astype(str)), axis=1))
all_social_matrix = social_bow.transform(features.apply(lambda x: ' '.join(x.astype(str)), axis=1))

# Transforming messages for training
train_social_matrix = social_bow.transform(social_train.apply(lambda x: ' '.join(x[['Gender', 'Age', 'EstimatedSalary']].astype(str)), axis=1))

# Transforming messages for testing
test_social_matrix = social_bow.transform(social_test.apply(lambda x: ' '.join(x[['Gender', 'Age', 'EstimatedSalary']].astype(str)), axis=1))

# Learning term weighting and normalization on the entire dataset
tfidf_transformer = TfidfTransformer().fit(all_social_matrix)

# Preparing TF-IDF for train data
train_tfidf = tfidf_transformer.transform(train_social_matrix)

# Preparing TF-IDF for test data
test_tfidf = tfidf_transformer.transform(test_social_matrix)

# Building Naïve Bayes model (using Multinomial Naïve Bayes for non-negative features)
classifier_nb = MultinomialNB()
classifier_nb.fit(train_tfidf, social_train['Purchased'])

# Predicting on test data
test_pred_nb = classifier_nb.predict(test_tfidf)

# Evaluating the model
accuracy_test_nb = accuracy_score(test_pred_nb, social_test['Purchased'])
print(f"Accuracy on test data: {accuracy_test_nb}")

