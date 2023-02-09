# Build-a-simple-linear-regression-model-using-statsmodels-and-SKLearn
# dataset(About Dataset)- https://www.kaggle.com/datasets/shivachandel/kc-house-data

# Classification
Identifying which category an object belongs to.

Applications: Spam detection, image recognition.

Algorithms: SVM, nearest neighbors, random forest,
# Regression
Predicting a continuous-valued attribute associated with an object.

Applications: Drug response, Stock prices.

Algorithms: SVR, nearest neighbors, random forest
# Clustering
Automatic grouping of similar objects into sets.

Applications: Customer segmentation, Grouping experiment outcomes

Algorithms: k-Means, spectral clustering, mean-shift,
# Dimensionality reduction
Reducing the number of random variables to consider.

Applications: Visualization, Increased efficiency

Algorithms: PCA, feature selection, non-negative matrix factorization
#Model selection
Comparing, validating and choosing parameters and models.

Applications: Improved accuracy via parameter tuning

Algorithms: grid search, cross validation, metrics,

# Preprocessing
Feature extraction and normalization.

Applications: Transforming input data such as text for use with machine learning algorithms.

Algorithms: preprocessing, feature extraction,


# build a simple linear regression model using both statsmodels and scikit-learn (sklearn) libraries in Python:

## Using statsmodels:

import statsmodels.api as sm
import pandas as pd

# load data
data = pd.read_csv('data.csv')

# prepare X and y variables
X = data[['x']]
y = data['y']

# add a constant to X to model the y intercept
X = sm.add_constant(X)

# fit the model
model = sm.OLS(y, X).fit()

# print model summary
print(model.summary())


##### Using scikit-learn:

import pandas as pd
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv('data.csv')

# prepare X and y variables
X = data[['x']].values
y = data['y'].values

# fit the model
model = LinearRegression().fit(X, y)

# print model coefficients
print('Intercept:', model.intercept_)
print('Slope:', model.coef_[0])

# ( Both of these models will give you the slope and y-intercept of the regression line. The statsmodels model will also provide additional statistical information such as p-values and confidence intervals for the coefficients.)


# 2nd way

# build a simple linear regression model using both statsmodels and scikit-learn (sklearn) in Python:


# Using statsmodels:
import statsmodels.api as sm
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")

# Define the independent and dependent variables
X = data["Independent_Variable"]
y = data["Dependent_Variable"]

# Add a constant to the independent variables
X = sm.add_constant(X)

# Build the model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())


# Using sklearn:

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")

# Define the independent and dependent variables
X = data["Independent_Variable"].values.reshape(-1, 1)
y = data["Dependent_Variable"].values

# Build the model
model = LinearRegression().fit(X, y)

# Print the coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])


# Note:( You'll need to replace "data.csv" with the path to your data file, and change the column names in the code to match the names in your data file.)



