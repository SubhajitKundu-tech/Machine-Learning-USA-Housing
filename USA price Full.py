import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

os.chdir(r"G:\$UBH@J!T\Data Science class Assingment\Class 7 Liniar Regration")
orton=pd.read_csv("USA_Housing Practice.csv")
orton.head()
pd.set_option("display.max_columns",None)

## show outlier

orton=orton.drop("Address",axis=1)
orton

def Subhajit (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data 

orton.boxplot(column=["Avg_Area_Income"])
plt.show()

orton=Subhajit(orton,"Avg_Area_Income")
orton

orton.boxplot(column=["Avg_Area_Income"])
plt.show()

orton.boxplot(column=["Avg_Area_House_Age"])
plt.show()

orton=Subhajit(orton,"Avg_Area_House_Age")
orton

orton.boxplot(column=["Avg_Area_House_Age"])
plt.show()

orton.boxplot(column=["Avg_Area_Number_of_Rooms"])
plt.show()

orton=Subhajit(orton,"Avg_Area_Number_of_Rooms")
orton

orton.boxplot(column=["Avg_Area_Number_of_Rooms"])
plt.show()

orton.boxplot(column=["Avg_Area_Number_of_Bedrooms"])
plt.show()

orton=Subhajit(orton,"Avg_Area_Number_of_Bedrooms")
orton

orton.boxplot(column=["Avg_Area_Number_of_Bedrooms"])
plt.show()

orton.boxplot(column=["Area_Population"])
plt.show()

orton=Subhajit(orton,"Area_Population")
orton

orton.boxplot(column=["Area_Population"])
plt.show()

orton.boxplot(column=["Price"])
plt.show()

orton=Subhajit(orton,"Price")
orton

orton.boxplot(column=["Price"])
plt.show()

## remove duplicates

orton.isnull().mean()*100
orton=orton.dropna()
orton
orton.isnull().mean()*100

## creat dummies

orton.head()

orton=pd.get_dummies(data=orton,columns=["Neighbourhood"],drop_first=True)
orton

orton.columns

## linier Regration model

kane=sm.ols(formula=
"""Price ~ Avg_Area_Income+Avg_Area_House_Age+Avg_Area_Number_of_Rooms+Avg_Area_Number_of_Bedrooms+
Area_Population+Neighbourhood_Rich+Neighbourhood_Super_Rich""",data=orton).fit()
kane.summary()

kane=sm.ols(formula=
"""Price ~ Avg_Area_Income+Avg_Area_House_Age+Avg_Area_Number_of_Rooms+
Area_Population+Neighbourhood_Rich+Neighbourhood_Super_Rich""",data=orton).fit()
kane.summary()
 
#predicting the outcomes

orton["pred"] = kane.predict()
orton.head()


var = pd.DataFrame(round(kane.pvalues,3))  #shows p value
kane.rsquared
var["coeff"] = kane.params  #coefficients

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = kane.model.exog    #.if I had saved data as rock

# this it would have looked like rock.model.exog

vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 
var["vif"] = vif
var

kane=sm.ols(formula=
"""Price ~ Avg_Area_Income+Avg_Area_House_Age+Avg_Area_Number_of_Rooms+
Area_Population+Neighbourhood_Rich""",data=orton).fit()
kane.summary()

kane=sm.ols(formula=
"""Price ~ Avg_Area_Income+Avg_Area_House_Age+Avg_Area_Number_of_Rooms+
Area_Population""",data=orton).fit()
kane.summary()

orton["pred"] = kane.predict()
orton.head()

var = pd.DataFrame(round(kane.pvalues,3))  #shows p value
kane.rsquared
var["coeff"] = kane.params  #coefficients

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = kane.model.exog    #.if I had saved data as rock

# this it would have looked like rock.model.exog

vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 
var["vif"] = vif
var

## mape

orton["mp"]=abs((orton["Price"]-orton["pred"])/orton["Price"])
(orton.mp.mean())*100

orton.head()


# assumption normality test
#Shapiro Wilk test
#Null Hypothesis: The residuals are normally distributed.
#Alternative Hypothesis: The residuals are not normally distributed.

from scipy import stats
stats.shapiro(kane.resid) 

#Checking for autocorrelation
#Null Hypothesis: Autocorrelation is absent.
#Alternative Hypothesis: Autocorrelation is present.

from statsmodels.stats import diagnostic as diag
diag.acorr_ljungbox(kane.resid , lags = 1)

#Checking heteroscedasticity
#Null Hypothesis: Error terms are homoscedastic
#Alternative Hypothesis: Error terms are heteroscedastic.

import statsmodels.stats.api as sms
from statsmodels.compat import lzip

#Breush-Pagan test:
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(kane.resid, kane.model.exog)
lzip(name, test)

orton.head()





































