# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:02:32 2024
@author: Vaibhav Bhorkade
"""
"""
Problem Statement :
A F&B manager wants to determine whether there is any significant difference 
in the diameter of the cutlet between two units. A randomly selected sample 
of cutlets was collected from both units and measured? Analyze the data and 
draw inferences at 5% significance level. Please state the assumptions and 
tests that you carried out to check validity of the assumptions.
"""

"""
Business Objective
Minimize : Minimize the variability in cutlet diameter between two units to ensure consistent quality across the F&B operation.
Maximize : To maintain cost-effectiveness in the measurement process.
Business Contraints -  Maintain cost-effectiveness in the measurement process to ensure feasibility and practicality.
"""

"""
Data Dictionary

  Feature        Type Relevance
0  Unit A  Continious  Relevant
1  Unit B  Continious  Relevant
"""
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageGrab
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

cutlets = pd.read_csv('Cutlets.csv')
cutlets.head(10)

cutlets.shape
# 51 rows and 2 columns

# 5 number summary
cutlets.describe()

# info
cutlets.info()

# check for null values
cutlets.isnull().sum()
# 16 null values in Unit A and Unit B

cutlets.dropna()

# Rename columns
cutlets.columns=['Unit_A','Unit_B']

cutlets.duplicated().sum()
# sum is 15, some duplicate values

# Plot
plt.subplots()
plt.subplot(121)
plt.boxplot(cutlets['Unit_A'])
plt.title('Unit_A')
plt.subplot(122)
plt.boxplot(cutlets['Unit_B'])
plt.title('Unit_B')
plt.show()

plt.subplots()
plt.subplot(121)
plt.hist(cutlets['Unit_A'], bins = 15)
plt.title('Unit_A')
plt.subplot(122)
plt.hist(cutlets['Unit_B'], bins = 15)
plt.title('Unit_B')
plt.show()

# displot
labels = ['Unit A', 'Unit B']
sns.distplot(cutlets['Unit_A'], kde = True)
sns.distplot(cutlets['Unit_B'],hist = True)
plt.legend(labels)

# QQ plot
sm.qqplot(cutlets["Unit_A"], line = 'q')
plt.title('Unit_A')
sm.qqplot(cutlets["Unit_B"], line = 'q')
plt.title('Unit_B')
plt.show()

statistic , p_value = stats.ttest_ind(cutlets['Unit_A'],cutlets['Unit_B'], alternative = 'two-sided')
print('p_value=',p_value)

from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

cutlets['Unit_A']=pd.DataFrame(mean_imputer.fit_transform(cutlets[['Unit_A']]))
cutlets.Unit_A.isna().sum()
cutlets['Unit_B']=pd.DataFrame(mean_imputer.fit_transform(cutlets[['Unit_B']]))
cutlets.Unit_A.isna().sum()
# H0=data is normal
# H1=data is not normal
print(stats.shapiro(cutlets.Unit_A))
# pvalue=0.07343>0.05, p is high null fly,hence data is normal
print(stats.shapiro(cutlets.Unit_B))
# pvalue=0.017<0.05, p is low,null go hence data is not normal
# Now let us apply Mann-Whitney Test
# H0=Diameters of cutlets are same
# H1=Diameters of cutlets are different 
scipy.stats.mannwhitneyu(cutlets.Unit_A,cutlets.Unit_B)
# pvalue=0.1790>0.05,p is high null fly,H0 is true
# Diameter of cutlets are same