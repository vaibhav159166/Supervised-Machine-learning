# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:51:44 2024
@author: Vaibhav Bhorkade
"""

"""
Telecall uses 4 centers around the globe to process customer order forms.
They audit a certain % of the customer order forms. Any error in order form
renders it defective and must be reworked before processing. The manager
wants to check whether the defective % varies by center. Please analyze
the data at 5% significance level and help the manager draw appropriate
inferences.
"""
"""
Business Objective
Minimize : To minimize defective across the all centers.
Maximize : To maximize the efficiency of order form.
Business Contraints -  5% significance level to ensure reliable conclusions.
"""

"""
Data Dictionary
   Feature           Type Relevance
0  Phillippines  Qualititative  Relevant
1     Indonesia  Qualititative  Relevant
2         Malta  Qualititative  Relevant
3         India  Qualititative  Relevant

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageGrab

# Load DataFrame
df=pd.read_csv("CustomerOrderform.csv")
df.head()

# info
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 315 entries, 0 to 314
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   Phillippines  300 non-null    object
 1   Indonesia     300 non-null    object
 2   Malta         300 non-null    object
 3   India         300 non-null    object
dtypes: object(4)
memory usage: 10.0+ KB

'''

# Check for NUll VAlue
df.isnull().sum()
# sum is 15 on every column
# Drop null value
df.dropna(inplace=True)

# Check all null values droped
df.isnull().sum()
'''
Phillippines    0
Indonesia       0
Malta           0
India           0
dtype: int64
'''

# 5 number summary
df.describe()

df.columns
'''
Index(['Phillippines', 'Indonesia', 'Malta', 'India'], dtype='object')
'''

# Value count
df.Phillippines.value_counts()
df.Indonesia.value_counts()
df.Malta.value_counts()
df.India.value_counts()

# Define the contingency table
contingency_table = pd.crosstab([df['Phillippines'], df['Indonesia'], df['Malta'], df['India']])

# Chi-square test of independence
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

# Check the data is balanced or imbalanced
print(df['Phillippines'].value_counts(),'\n',df['Indonesia'].value_counts(),'\n',df['Malta'].value_counts(),'\n',df['India'].value_counts())

# Calculate the Contingeny table
contingency_table = [[271,267,269,280],
                    [29,33,31,20]]

print(contingency_table)

stat, p, df, exp = stats.chi2_contingency(contingency_table)
print("Statistics = ",stat,"\n",'P_Value = ', p,'\n', 'degree of freedom =', df,'\n', 'Expected Values = ', exp)

# Defining Expected values and observed values
observed = np.array([271, 267, 269, 280, 29, 33, 31, 20])
expected = np.array([271.75, 271.75, 271.75, 271.75, 28.25, 28.25, 28.25, 28.25])

# Compare Evidences with Hypothesis using t-statictic

test_statistic , p_value = stats.chisquare(observed, expected, ddof = df)
print("Test Statistic = ",test_statistic,'\n', 'p_value =',p_value)

