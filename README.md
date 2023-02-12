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

# Steps involved:

1. Import libraries

2. Split the data for training and testing

3. Rescaling the features

4. Add a constant and instantiate the object

5. Fit the model

6. Predict the model

7. Evaluate the model



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

# This very simple case-study is designed to get you up-and-running quickly with statsmodels. Starting from raw data, we will show the steps needed to estimate a statistical model and to draw a diagnostic plot. We will only use functions provided by statsmodels or its pandas and patsy dependencies.

Loading modules and functions¶
After installing statsmodels and its dependencies, we load a few modules and functions:

In [1]: import statsmodels.api as sm

In [2]: import pandas

In [3]: from patsy import dmatrices
pandas builds on numpy arrays to provide rich data structures and data analysis tools. The pandas.DataFrame function provides labelled arrays of (potentially heterogenous) data, similar to the R “data.frame”. The pandas.read_csv function can be used to convert a comma-separated values file to a DataFrame object.

patsy is a Python library for describing statistical models and building Design Matrices using R-like formulas.

# Note

This example uses the API interface. See Import Paths and Structure for information on the difference between importing the API interfaces (statsmodels.api and statsmodels.tsa.api) and directly importing from the module that defines the model.

Data¶
We download the Guerry dataset, a collection of historical data used in support of Andre-Michel Guerry’s 1833 Essay on the Moral Statistics of France. The data set is hosted online in comma-separated values format (CSV) by the Rdatasets repository. We could download the file locally and then load it using read_csv, but pandas takes care of all of this automatically for us:

In [4]: df = sm.datasets.get_rdataset("Guerry", "HistData").data
The Input/Output doc page shows how to import from various other formats.

We select the variables of interest and look at the bottom 5 rows:

In [5]: vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']

In [6]: df = df[vars]

In [7]: df[-5:]
Out[7]: 
      Department  Lottery  Literacy  Wealth Region
81        Vienne       40        25      68      W
82  Haute-Vienne       55        13      67      C
83        Vosges       14        62      82      E
84         Yonne       51        47      30      C
85         Corse       83        49      37    NaN
Notice that there is one missing observation in the Region column. We eliminate it using a DataFrame method provided by pandas:

In [8]: df = df.dropna()

In [9]: df[-5:]
Out[9]: 
      Department  Lottery  Literacy  Wealth Region
80        Vendee       68        28      56      W
81        Vienne       40        25      68      W
82  Haute-Vienne       55        13      67      C
83        Vosges       14        62      82      E
84         Yonne       51        47      30      C


## Difference between SKlearn and Statsmodels

# Introduction

One of the benefits to programming in Python is the vast community and universe of libraries they have created. Those attempting to create linear models in Python will find themselves spoiled for choice. Two of the most popular linear model libraries are scikit-learn’s linear_model and statsmodels.api. While both libraries can successfully fit linear models and calculate predictions, they differ in several ways and are not necessarily means to the same end. In general, scikit-learn is designed for prediction, while statsmodels is more suited for explanatory analysis. In this article, I explore several of the key differences between these two libraries and compare how they calculate ordinary least squares (OLS), their efficiencies, the features they offer, how they perform logistic regressions, and the other linear models they are capable of modeling.


It would not take long for a Python programmer to notice the different syntaxes between the two libraries. In my opinion, scikit-learn’s syntax is more intuitive to someone familiar with object-oriented programming, though statmodels’ syntax is certainly functional. The purpose of this article is to examine these modules in greater depth. Those interested in syntax should examine the documentations

## A key difference between the two libraries is how they handle constants. Scikit-learn allows the user to specify whether or not to add a constant through a parameter, while statsmodels’ OLS class has a function that adds a constant to a given array.

Scikit-learn’s LinearRegression class is, per the docstring in the source code, “plain [OLS] (scipy.linalg.lstsq) wrapped as a predictor object.” SciPy computes the solution to the least-squares equation Ax = b. Meanwhile, statsmodels’ OLS class provides two algorithms, chosen by the attribute “methods”: the Moore-Penrose pseudoinverse, the default algorithm and similar to SciPy’s algorithm, and QR factorization. All of these algorithms ultimately return the same result in most cases. What is likely more interesting to most programmers is how quickly each class arrives at this result.

# While scikit-learn is slightly faster than statsmodels for 1,000 or less observations, this difference is not significant per the t-test analysis. Sci-kit learn is significantly faster for datasets with more than 1,000 observations. On my computer, for 108 observations, the largest power of 10 I could reasonably test, scikit-learn is about two and half times as fast as statsmodels. In performing these tests, I did not make full use of scikit-learn’s n_jobs parameter which allows the user to use additional cores to potentially fit models and make predictions even faster. Ultimately, I concluded that scikit-learn was faster than statsmodels at fitting ordinary least squares regressions. The code for the experiment is available in the accompanying Github repository under time_tests.py, while the experiment is carried out in sklearn_statsmodels_time_comp.ipynb.

# In general, scikit-learn’s linear models, such as ridge and lasso regressions, are suitable for regularization and prediction. They are designed to prevent the model from overtraining to ultimately provide more accurate predictions. Scikit-learn also has other modules that assist in calculating prediction metrics and cross-validation metrics, both useful tools for prediction. On the other hand, the linear models offered in statsmodels are meant to be implemented when an underlying assumption of OLS is broken. Weighted least squares (WLS), for example, helps correct for heteroskedasticity. These models are useful when performing rigorous statistics.



# In general, scikit-learn is designed for machine-learning, while statsmodels is made for rigorous statistics. Both libraries have their uses. Before selecting one over the other, it is best to consider the purpose of the model. A model designed for prediction is best fit using scikit-learn, while statsmodels is best employed for explanatory models. To completely disregard one for the other would do a great disservice to an excellent Python library. I, for one, have room in my heart for more than one linear regression library. Hopefully, all of you do too.
