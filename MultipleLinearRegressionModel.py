# Multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, :5].values
y = dataset.iloc[:, 6].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() #label encoder will simply apply integer values to different categories all in single column
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [1, 4]) #OneHotEncoder will convert the categorical column into dummy encoding(see notes)
X = onehotencoder.fit_transform(X).toarray()

X = X[:, [1, 3, 4, 5, 6]]

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1338, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #The lower the p value, the more significant is the feature

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Accuracy of Train set prediction
y_predTrain = regressor.predict(X_train)
from sklearn.metrics import r2_score
r2_Score_Train = r2_score(y_train, y_predTrain) * 100

# Accuracy of Test set prediction
y_predTest = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_Score_Test = r2_score(y_test, y_predTest) * 100