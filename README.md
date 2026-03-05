# Machine Learning Project – USA Housing Price Prediction

## Project Overview

This project builds a **Linear Regression model** to predict housing prices in the USA using several housing features such as income, house age, number of rooms, and population.

The objective is to understand which factors influence house prices and build a predictive model using Python.

---

## Dataset

The dataset contains housing information with the following variables:

* Avg_Area_Income – Average income of people in the area
* Avg_Area_House_Age – Average age of houses in the area
* Avg_Area_Number_of_Rooms – Average number of rooms in houses
* Avg_Area_Number_of_Bedrooms – Average number of bedrooms
* Area_Population – Population of the area
* Neighbourhood – Type of neighborhood (Normal / Rich / Super Rich)
* Price – House price (Target Variable)

---

## Project Workflow

### 1. Data Cleaning

* Removed unnecessary column **Address**
* Checked for missing values
* Removed null values using `dropna()`

### 2. Outlier Detection and Treatment

Outliers were detected using the **Interquartile Range (IQR) method**.

A custom function was created to remove extreme values from variables such as:

* Avg_Area_Income
* Avg_Area_House_Age
* Avg_Area_Number_of_Rooms
* Avg_Area_Number_of_Bedrooms
* Area_Population
* Price

Boxplots were used before and after removing outliers.

### 3. Feature Engineering

The categorical variable **Neighbourhood** was converted into dummy variables using:

`pd.get_dummies()` with `drop_first=True`.

Example:

Neighbourhood_Rich
Neighbourhood_Super_Rich

---

## Model Building

A **Linear Regression model** was built using the **statsmodels OLS (Ordinary Least Squares)** method.

Initial model included all variables:

Price ~ Avg_Area_Income + Avg_Area_House_Age + Avg_Area_Number_of_Rooms + Avg_Area_Number_of_Bedrooms + Area_Population + Neighbourhood variables

Insignificant variables were removed based on **p-values and VIF** to improve model performance.

Final model features:

* Avg_Area_Income
* Avg_Area_House_Age
* Avg_Area_Number_of_Rooms
* Area_Population

---

## Model Evaluation

Model performance was evaluated using:

* R-squared
* p-values
* Variance Inflation Factor (VIF)
* Mean Absolute Percentage Error (MAPE)

MAPE was calculated using:

|Actual Price - Predicted Price| / Actual Price

---

## Assumption Testing

The following regression assumptions were tested:

### Normality Test

Shapiro-Wilk Test was used to check whether residuals are normally distributed.

### Autocorrelation Test

Ljung-Box Test was applied to check autocorrelation in residuals.

### Heteroscedasticity Test

Breusch-Pagan Test was used to verify constant variance of residuals.

---

## Results and Insights

* **Area Income** has a strong positive impact on housing prices.
* **Number of Rooms** significantly increases house prices.
* **Area Population** also contributes to price variation.
* Removing insignificant variables improved model stability.

The model achieved good predictive performance with acceptable error levels.

---

## Tools and Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Statsmodels
* SciPy

---

## Conclusion

A Linear Regression model was successfully built to predict housing prices using housing features.

After cleaning the data, removing outliers, and validating regression assumptions, the model provided meaningful insights into factors affecting housing prices.

---

## Author

Subhajit Kundu
Aspiring Data Scientist


