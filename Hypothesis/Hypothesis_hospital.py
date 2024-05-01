# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:22:12 2024
@author: Vaibhav Bhorkade
"""
"""
A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports 
of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports of 4 
laboratories. TAT is defined as sample collected to report dispatch. Analyze the data and determine whether there
is any difference in average TAT among the different laboratories at 5% significance level.
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats

# Load DataFrame
df=pd.read_csv("lab_tat_updated.csv")
df.head()

# Rename Columns
df.columns=["Lab_1","Lab_2","Lab_3","Lab_4"]

df.isna().sum()
# Sum is 0, No null values

df.dtypes
'''
Lab_1    float64
Lab_2    float64
Lab_3    float64
Lab_4    float64
dtype: object
'''
# All values in float

df.duplicated()
df.duplicated().sum()
# No duplicate value

# 5 number summary
df.describe()

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Plotting QQ plot to check whether the distribution follows normal distribution or not
sm.qqplot(df['Lab_1'], line = 'q')
plt.title('Laboratory 1')

sm.qqplot(df['Lab_2'], line = 'q')
plt.title('Laboratory 2')

sm.qqplot(df['Lab_3'], line = 'q')
plt.title('Laboratory 3')

sm.qqplot(df['Lab_4'], line = 'q')
plt.title('Laboratory 4')
plt.show()

# Displot
plt.figure()
labels = ['Lab 1', 'Lab 2','Lab 3', 'Lab 4']
sns.distplot(df['Lab_1'], kde = True)
sns.distplot(df['Lab_2'],hist = True)
sns.distplot(df['Lab_3'],hist = True)
sns.distplot(df['Lab_4'],hist = True)
plt.legend(labels)

# Now let us check the normality 
# H0=data is normal
# H1=Data is not normal
stats.shapiro(df.Lab_1)
# pvalue=0.4231 >0.05 hence data is normal
stats.shapiro(df.Lab_2)
# pvalue=0.8637 >0.05 hence data is normal
stats.shapiro(df.Lab_3)
# pvalue=0.0654 >0.05 hence data is normal
stats.shapiro(df.Lab_4)
# pvalue=0.6618 >0.05 hence data is normal
# All the columns are normal

scipy.stats.levene(df.Lab_1,df.Lab_2,df.Lab_3,df.Lab_4)
# pvalue=0.3810
# hence p high null fly
# H0:All the labs are having equal TAT
# H1:At least one is having different TAT

F,pval=stats.f_oneway(df.Lab_1,df.Lab_2,df.Lab_3,df.Lab_4)
pval
# 2.143740909435053e-58 .
# p low null go
# H1 is true

