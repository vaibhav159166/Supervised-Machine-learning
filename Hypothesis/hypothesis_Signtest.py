# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:31:33 2024
@author: Vaibhav Bhorkade
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats

import statsmodels.stats.descriptivestats as sd
import statsmodels.stats.weightstats

# 1 sample

marks=pd.read_csv("Signtest.csv")
# Normal QQ plot
import pylab
stats.probplot(marks.Scores,dist="norm",plot=pylab)
# Data is not normal
# H0 - data is normal
# H1- data is not normal
stats.shapiro(marks.Scores)

# p_value is 0.024 > 0.005 , p is high null fly
# Decision : data is not normal

# Let us check the distribution of the data
marks.Scores.describe()
# 1 sample sign test
sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
# p_value is 0.82> 0.05 so p is high null fly
# Decision: 
# H0=scores are either equal or less than 80

import matplotlib.pyplot as plt 
plt.boxplot(marks.Scores)

# Non-Parametric Test--> 1 sample sign test






###### 1 sample Z test ############
#importing the data
fabric=pd.read_csv("Fabric_data.csv")

#calculating the normality test
print(stats.shapiro(fabric))
#0.1450 > 0.005 means H0 is True
#calculating the mean
np.mean(fabric)

# Z test
#parameters in z test , value is  mean of data
from statsmodels.stats import weightstats as stests

ztest,pval=stests.ztest(fabric, x2=None, value=158)

print(float(pval))



##################mann-Whitney Test###################
import pandas as pd
import numpy as np
import scipy
from scipy import stats 


# from statmodels.stats import weightstats sa stests
import statsmodels.stats.weightstats
##################mann-Whitney Test###################
#vechical with and without addictive
#H0= fuel additive does not impact the performance
#H1= fuel additive does impact the performance
fuel=pd.read_csv()
fuel

#rename the column
fuel.columns="Without_additive","With_additive"

#normailty -test
#H0=data is normal

print(stats.shapiro(fuel.Without_additive))     #p high null fill
print(stats.shapiro(fuel.With_additive))        #p low null go

#without_additive is normal
# with additive is not normal
# when two sample are not normal then mannwhitney test
#non-parametric test case'
# mann-withney test
scipy.stats.mannwhitney(fuel.Without_additive,fuel.With_additive)

#p-value =0.4457>0.05 so p is high null fly 
#H0= fuel additive does not impact the performance


################  Paired T-test    ########################
# When two features are normal then paired T test
# A univariaate test what tests for a significant between 2relative
sup=pd.read_csv("paired2.csv")
sup.describe()
# H0 
# H1
# Normality Test
stats.shapiro(sup.SupplierA)
stats.shapiro(sup.SupplierB)

import seaborn as sns
sns.boxplot(data=sup)

ttest,pval=stats.ttest_rel(sup["SupplierA"],sup["SupplierB"] )
print(pval)


# p-value = 0 < 0.05 so p low null go


# # # # # # # # # # # # # #  # # # #  ## # # # # # # # # # # # 
# 2 sample T test

# load the data 
prom=pd.read_excel("Promotion.xlsx")
# H0 : InterestRateWaive < StandardPromotion
# H1 : InterestRateWaive > StandardPromotion

prom.columns="InterestRateWaive","StandardPromotion"

# Normality test
stats.shapiro(prom.InterestRateWaive)

print(stats.shapiro(prom.StandardPromotion))

# Data are normal

# Varience test
help(scipy.stats.levene)
# H0 = Both columns have equal variance
# H1 = Both columns have not equal variance

scipy.stats.levene(prom.InterestRateWaive,prom.StandardPromotion)
# p- value = 0.28 > 0.05 so p high nll fly => Equal variances

# 2 sample T test
scipy.stats.ttest_ind(prom.InterestRateWaive,prom.StandardPromotion)
# H0 : equal 
# H1 : unequal
# p value - 0.024 < 0.05 so p low null go


################################################################

# one -way ANOVA

"""
Contract Renewal
A marketing organization outsources thier back-office operations to three diffrent 
suppliers. The contracts are up for renewal and the cmo wants to determine wheater 
they should renew contracts with all suppliers or any specific suppleirs.
CMO want to renew the contract of supplirs ....
"""

con_renewal=pd.read_excel("ContractRenewal_Data(unstacked).xlsx")

con_renewal
con_renewal.columns="SupplierA","SupplierB","SupplierC"
# H0 = All the 3 suppliers have equal mean transaction time
# H1 = All the 3 suppliers have not equal mean transaction time
# Normality Test
stats.shapiro(con_renewal.SupplierA)
# P value = 0.89 > 0.005 SupplierA is normal
stats.shapiro(con_renewal.SupplierB)
# Shapiro test
stats.shapiro(con_renewal.SupplierC)

help(scipy.stats.levene)

scipy.stats.levene(con_renewal.SupplierA,con_renewal.SupplierB,con_renewal.SupplierC)
# The levene test tests the null hypothesis 
# that all input samples are form populations with equal variances
# pvalue=0.77 > 0.005 , p is high null fly
# H0= All input sample equal variance

# One way ANOWA
F,p=stats.f_oneway(con_renewal.SupplierA,con_renewal.SupplierB,con_renewal.SupplierC)

p

# p high null fly
# All three suppliers have equal mean transation time

# # # # # # # # #  # # # # #  # # # # # #  # # # # # # # #  # # ## 
 
# 2 - Proportion test
import numpy as np

"""
Johnnie Talkers soft drinks division sales manager has been planning to 
launch a new sales incentive program for their sales executives. The sales
executives felt that adults (>40 yrs) wont buy , children will & hence 
requested sales manager not to launch the program . Anlayze...
"""
two_prop_test=pd.read_excel("JohnyTalkers.xlsx")

from statsmodels.stats.proportion import proportions_ztest

tab1=two_prop_test.Person.value_counts()
tab1

tab2=two_prop_test.Drinks.value_counts()
tab2

# crosstable table
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)

count=np.array([58,152])
nobs=np.array([480,740])

stats,pval=proportions_ztest(count, nobs,alternative="two-sided")
print(pval)
# Pvalue - 0.000

stats, pval=proportions_ztest(count, nobs,alternative="larger")
print(pval)
# Pvalue - 0.999

# # # # #  # # #  # # # #  # # #  # #  # #  # # # # # #  ## # # #  ## #  # ## 

# Chi- Square Test

Bahaman=pd.read_excel("Bahaman.xlsx") 
Bahaman

count=pd.crosstab(Bahaman["Defective"],Bahaman["Country"])
count
Chisquares_results=scipy.stats.chi2_contingency(count)

Chi_square=[['Test Statistic', 'p-value'],[Chisquares_results[0],Chisquares_results[1]]]
Chi_square

'''
you use chi2_contigency when you want to test
wheather two (or more )groups have the same distribution.
'''
# H0 = Null hypothesis : the two groups have
     # no significant diffrence.
# since p=0.63> 0.05 Hence H0 is true

















