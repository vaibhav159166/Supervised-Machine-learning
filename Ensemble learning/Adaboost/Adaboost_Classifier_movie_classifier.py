# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:14:37 2024
@author: Vaibhav Bhorkade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.metrics import accuracy_score,classification_report

data=pd.read_csv('movies_classification.csv')

# Data information
data.head()
data.info()
data.isna().sum()
# EDA
target=data["Start_Tech_Oscar"]
sns.countplot(x=target,palette='winter')
plt.xlabel("Oscar Rate");
# Our data is evenly distributed. Atleast 200 are there in both choice
plt.figure(figsize=(16,8))
sns.heatmap(data.corr(), annot=True,cmap='YlGnBu',fmt='.2f')
# Observations

# 1) Lead_Actor_Rating, Lead_Actress_rating, Director_rating and Producer

sns.countplot(x='Genre',data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title('O chance based on Ticket Class', fontdict=10)

# Observations
# Here are more chances of getting Oscar in Drama , Comedy and Action Genre
sns.countplot(x='3D_available',data=data, hue='Start_Tech_Oscar',palette='pastel')
plt.title('O chance based on Ticket class',fontsize=10);
# Observations
# It is clear from the plot that if 3D is available the there is chance 
sns.set_context('notebook',font_scale=1.2)
fig,ax=plt.subplot(2, figsize=(20,13))

plt.subtittle('Distribution of Twitter_hastages and collection based on target variable',fontsize=20)

ax1=sns.histplot(x="Twitter_hastags",data=data,hue='Start_Tech_Oscar',kde=True,ax=ax[0],palette='winter')

ax1.set(x_label='Twitter_hastages',title='Distribution of Twitter_hastages based on target variable')

ax2=sns.histplot(x="Collection",data=data,hue='Start_Tech_Oscar',kde=True,ax=ax[0],palette='viridis')

ax2.set(x_label='Collection',title='Distribution of Fare based on target variable')

plt.show()

data.hist(bins=30, figsize=(20,15),color='#005b96')

# As we can see there are outliers in Twitter_hastags, Marketing_expense, Time_taken
sns.boxplot(x=data['Twitter_hastags'])
sns.boxplot(x=data['Marketing expense'])
sns.boxplot(x=data['Time_taken'])
sns.boxplot(x=data['Avg_age_actors'])
# Winsorizor
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Twitter_hastags']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

data1=winsor.fit_transform(data[['Twitter_hastags']])

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Time_taken']
                  )
dat=winsor.fit_transform(data[['Time_taken']])

sns.boxplot(data[['Twitter_hastags']])
sns.boxplot(data1['Twitter_hastags'])

data['Twitter_hastags']=data1['Twitter_hastags']
data['Time_taken']=dat['Time_taken']
# Checking skewness
skew_df=pd.DataFrame(data.select_dtypes(np.number).columns,columns=['Feature'])
skew_df['Skew']=skew_df['Feature'].apply(lambda feature: skew(data[feature]))
skew_df['Absolute Skew']=skew_df['Skew'].apply(abs)
skew_df['Skewed']=skew_df["Absolute Skew"].apply(lambda x:True if x>=0.5 else False)
skew_df
# Total Charges column is clearly skwed as we also saw in the histogram
for column in skew_df.query("Skewed == True")['Feature'].values:
    data[column]=np.log1p(data[column])
    
data.head()
# Encoding
data1=data.copy()
data1=pd.get_dummies(data1)

data1.head()
# Scaling
data2=data1.copy()
sc=StandardScaler()
data2[data1.select_dtypes(np.number).columns]=sc.fit_transform(data2[data1.select_dtypes(np.number).columns])
data2.drop(["Start_Tech_Oscar"],axis=1,inplace=True)
data2.head()

#Spliting
data_f=data2.copy()
target=data['Start_Tech_Oscar']
target=target.astype(int)
target

X_train,X_test,y_train,y_test=train_test_split(data_f, target,test_size=0.2,stratify=target,random_state=42)

# Modeling

from sklearn.ensemble import AdaBoostClassifier

ada_clf=AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)

ada_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(X_test))
accuracy_score(y_test, ada_clf.predict(X_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(X_train))

from sklearn.metrics import accuracy_score,confusion_matrix

