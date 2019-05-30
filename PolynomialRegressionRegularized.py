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

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X) 
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1, 1))

#Fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 0)

#Fitting Lasso regression 
lasso_regressor = Lasso(alpha=0.01, fit_intercept=False)
lasso_regressor.fit(X_train, y_train)

# Predicting the Train set results
y_predTest = lasso_regressor.predict(X_test)
y_predTest = sc_y.inverse_transform(y_predTest)

# Accuracy of Train set prediction
y_predTrain = lasso_regressor.predict(X_train)
y_predTrain = sc_y.inverse_transform(y_predTrain)
y_train = sc_y.inverse_transform(y_train)
from sklearn.metrics import r2_score
r2_Score_Train = r2_score(y_train, y_predTrain) * 100

# Accuracy of Test set prediction
y_predTest = lasso_regressor.predict(X_test)
y_predTest = sc_y.inverse_transform(y_predTest)
y_test = sc_y.inverse_transform(y_test)
from sklearn.metrics import r2_score
r2_Score_Test = r2_score(y_test, y_predTest) * 100