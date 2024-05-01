# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:02:37 2024
@author: Vaibhav Bhorkade
"""
"""
Fantaloons Sales managers commented that % of males versus females walking 
into the store differ based on day of the week. Analyze the data and determine 
whether there is evidence at 5 % significance level to support this hypothesis. 
"""

"""
Data Dictionary
Features    Type
Weekdays nominal data
Weekend  nominal data
"""
import pandas as pd
import scipy.stats as stats

# Load DataFrame
df=pd.read_csv("Fantaloons.csv")
df.head()

df.tail()

# Check for missing values
df.isnull().sum()
'''
Weekdays    25
Weekend     25
dtype: int64
'''

# Drop Null values
df.dropna(inplace=True)
# Check null value droped or not

df.isnull().sum()
# sum is 0.

df.info()
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 400 entries, 0 to 399
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Weekdays  400 non-null    object
 1   Weekend   400 non-null    object
dtypes: object(2)
memory usage: 9.4+ KB
'''

# 5 number summary
df.describe()

# Contingency table
contingency_table = pd.crosstab(df['Weekdays'], df['Weekend'])
contingency_table

# Chi-square test
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
# Chi-square Test
print("Chi-square statistic:", chi2_stat)
print("p-value:", p_val)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)

# There is no evidence to reject the null hypothesis. 

