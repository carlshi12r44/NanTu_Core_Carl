import intercepts
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def handler(func, *args, **kwargs):
    model = func(*args, **kwargs)
    print(f"in intercepts I am receiving the coefficients from model {model} as {model.coef_}") # this return as tuple
    print(f"in intercepts I am receiving the intercept from model {model} as {model.intercept_}")
    print(f"in intercepts I am receiving function output as {func(*args, **kwargs)}")

if __name__=="__main__":
    ## example of sklearn
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # register through intercepts 
    intercepts.register(regr.fit, handler)
    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    intercepts.unregister_all()
    print(f"in main I have coefficients for linear regression as {regr.coef_}")