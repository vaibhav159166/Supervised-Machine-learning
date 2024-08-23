# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:06:31 2024
@author: Vaibhav Bhorkade
"""
# multiple correlation regression analysis
import pandas as pd
import numpy as np
import seaborn as sns
cars=pd.read_csv("Cars.csv")
# Exploratory Data Analysis
# 1. Measure the central tendency
# 2. Measure the dispersion
# 3. Third moment business decision - Skewness
# 4. Fouth moment business decision - Kurtosis
# 5. Probablity distribusion
# 6. Graphical representation (Histogram, Boxplot)
cars.describe()
# Graphical representation 
import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,82,1))
sns.distplot(cars.HP)
# data is right skewed
plt.boxplot(cars.HP)
# There are sevral outliers in HP column
# Similar oerations are expected for other three columns
sns.distplot(cars.MPG)
# data is slightly left disributed
plt.boxplot(cars.MPG)
# There are no outliers
sns.distplot(cars.VOL)
# data slightly left disributed
plt.boxplot(cars.VOL)
# There are sevral outliers
sns.distplot(cars.WT)
plt.boxplot(cars.WT)
# There are several outliers
# Now let us plot joint plot , joint plot is to show scatter
# histogram
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])

# Now let us plot the count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
# Countplot shows how many times the each value occure
# 92 HP value occured 7 Times

# QQ plot
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist="norm",plot=pylab)
plt.show()
# MPG data is normally distributed
# There are 10 scatter plots need  to plotted, one by to plot , 
# so we can use pair plots
sns.pairplot(cars.iloc[:,:])
# Linearity : moderate , direction : positive , strength : average

# You can check the collinearity problem between the input
# you can check between between HP and SP , they are strongly 
# same way you can check WT and VOL

# now let us check r value between variables
cars.corr()
# you can check SP and HP, r value is 0.97 and same way
# you can check WT and VOL , it got 0.999
# which is greater
# Now althrough we observed strongly correlated pairs
# still we will go for linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
# R square value observed 0.771 < 0.85
# p-values of WT and VOL is 0.814 and 0.556 which is very 
# it means it is greater than 0.05 , WT and VOL columns
# we need to ignore
# or delete . Instead deleting 81 entries,
# let us check row wise outliers
# identifying is there any influential value
# To check you can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# 76 is the data the value which has got outliers
# go to data frame and check 76 th entry
# let us delete that entry
cars_new=cars.drop(cars.index[[76]])

# again apply regression to cars_new
ml_new=smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new.summary()
# R-square value is 0.819 but p values are same
# Now next option is delete the column but
# question is which column is to be deleted
# we have already checked correlation factor r
# VOL has got -0.529 

# another approach is to check the collinearity
# rsquare is giving that values
# we will have to apply regression w.r.t.x1 and input
# as x2,x3 and x4 so on so forth
rsq_hp=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp
# VIF is variance influential factor, calculating VIF helps of X1 w.r.t. X2,X3 and X4
rsq_wt=smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt=1/(1-rsq_wt)
vif_wt

rsq_vol=smf.ols('VOL~HP+WT+SP',data=cars).fit().rsquared
vif_vol=1/(1-rsq_vol)
# vif_wt = 639.53 and Vif_vol = 638.80 hence vif_wt
# is greater , thumb rule is vif should not be greater than 1
rsq_sp=smf.ols('SP~HP+VOL+WT',data=cars).fit().rsquared
vif_sp=1/(1-rsq_sp)
vif_sp

# storing the values in dataframe
d1={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame

# Let us drop WT and apply correlation to remailing three
final_ml=smf.ols('MPG~VOL+SP+HP',data=cars).fit()
final_ml.summary()
# R square is 0.770 and P values 0.00,0.012, < 0.05

# prediction
pred=final_ml.predict(cars)

## QQ plot
res=final_ml.resid
sm.qqplot(res)
plt.show()
# This is on residual which is obtained on training 
# errors are obtained on test data
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

# let us plot the residual plot , which takes the residuals 
sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residuals')
plt.title('FItted VS Residual')
plt.show()
# residuals plots are used

# spliting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train,cars_test=train_test_split(cars,test_size=0.2)
# preparing the model on train data
model_train=smf.ols('MPG~VOL+SP+HP',data=cars_train).fit()
model_train.summary()
test_pred=model_train.predict(cars_test)

test_error=model_train.predict(cars_test)
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse
