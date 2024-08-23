# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:47:52 2024
@author: Vaibhav Bhorkade
"""
import pandas as pd
import numpy as np
import seaborn as sns

wcat=pd.read_csv("wc-at.csv")
# Exploratory data analysis
# 1. Measure the central tendancy
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
wcat.info()
wcat.describe()

# Graphical Representation
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1, 110,1 ))
sns.displot(wcat.AT,kde=True)
# Data is normal but right skwed
plt.boxplot(wcat.AT)
# data is right skwed
# scatter plot
plt.scatter(x=wcat["Waist"],y=wcat['AT'],color='green')
# Linealy scattered
# direction:positive,linearity:moderate,Strength:poor
# Now let us calculate correlation coeficient
np.corrcoef(wcat.Waist,wcat.AT)
# Let us check direction using cover factor
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
# now let us apply to linear regression model
import statsmodels.formula.api as smf
# All machine learning algo. are implement using sklearnbut for this statsmodel
# Package is being used because it gives you 
# backend calculations of bita -0 and bita-1
model=smf.ols('AT~Waist',data=wcat).fit()
model.summary()
# ols helps to find best fit model, which causes least square error
# first you check R squared value = 0.670 , if R square = 0.8
# means that model is best fit
# fit , if R - Square =0.8 to 0.6 moderate fit
# Next you check P>|t|=0 , it means less than alpha
# alpha is 0.05 , Hence the model is accepted


# Regression line
pred1=model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,"r")
plt.show()

## error calculations
res1=wcat.AT-pred1
np.mean(res1)
# It must be zero and hence it 10^ -14=~0
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
# 32.76 lesser the value better the model
# how to imporove this model , transformation of
plt.scatter(x=np.log(wcat['Waist']),y=wcat['AT'],color='brown')
np.corrcoef(np.log(wcat.Waist),wcat.AT)
# r value is 0.82 < 0.85 hence moderate linearity
model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.summary()
# Again check the R-square value = 0.67 which is less than 0.8
# P value is 0 less than 0.05
pred2=model2.predict(pd.DataFrame(wcat['Waist']))
# check  wcat and pred2 from variable explorer
## Scatter diagram
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.scatter(np.log(wcat.Waist),pred2,"r")
plt.legend(['Atual data','Predicated data'])
## Error calculations
res2=wcat.AT-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

# there no considerable changes
# Now let us change y value instead of x
plt.scatter(x=wcat['Waist'], y=np.log(wcat['AT']),color='orange')
np.corrcoef(wcat.Waist,np.log(wcat.AT))
# r value is 0.84 < 0.85 hence moderate linearity
model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.summary()
# Again check the R-square value = 0.7 which is less than 0.8
# P value is 0.02 less than 0.05
pred3=model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at=np.exp(pred3)
# cgeck wcat and pred3_ at for variable explorer
## Scatter diagram
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,"r")
plt.legend(['Atual data','predicated data'])
plt.show()
# Error calculations
res3=wcat.AT-pred3_at
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
# RMSE is 38.53
# P= 0.0 < 0.05
# bita - 0 = -7.82
# bita - 1 = 0.22
####################################################
# Polynominal transformation
# Now let us make Y=log(AT) abd X=Waist,X*X=Waist.Waist
# Here r can not be calculated
model4=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
model4.summary()
# R-squared - 0.779
pred4=model4.predict(pd.DataFrame(wcat['Waist']))
pred4
pred4_at=np.exp(pred4)
pred4_at

# Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicated line','Observed data_model3'])
plt.show()
##################
# error calculations
res4=wcat.AT-pred4_at
res_sqr=res4*res4
rmse4=np.mean(res_sqr)
rmse4
# 32.24
# Among all the models , model4 is the best
#############################################################
data={'model':pd.Series(['SLR','Log_model','Exp_model','Poly_model'])}
data
table_rmse=pd.DataFrame(data)
table_rmse

# We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(wcat, test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
# Y is log(AT) and X=Waist
final_model.summary()
# R-squared =  0.775 < 0.85 , there is scope of improvement
# p=0.000<0.05 hence acceptable
# bita-0 = -7.8241
# bita-1 = 0.2289
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at

####
train_pred=final_model.predict(pd.DataFrame(train))

train_pred_at=np.exp(train_pred)
train_pred_at

# Evolution on test data
test_err=test.AT-test_pred_at
test_sqr=test_err*test_err
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse

# Evoluation on train data
train_res=train.AT-train_pred_at
train_sqr=train_res*train_res
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse

# This is an Overfit model
# test_rmse > train_rmse