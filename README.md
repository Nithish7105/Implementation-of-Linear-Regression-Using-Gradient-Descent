# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:

```
/*
Program to implement the linear regression using gradient descent.
Developed by: NITHISH KUMAR.B
RegisterNumber:  212223040134
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def linear_regression(X, y, iters=1000, learning_rate=0.01):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept term
    theta = np.zeros((X.shape[1], 1))
    
    for _ in range(iters):
        predictions = X.dot(theta)
        errors = predictions - y.reshape(-1, 1)
        gradient = (1 / X.shape[0]) * X.T.dot(errors)
        theta -= learning_rate * gradient
    
    return theta

data = pd.read_csv('50_Startups.csv', header=0)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [3])  
], remainder='passthrough')

X = ct.fit_transform(X)

y = y.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

theta = linear_regression(X_scaled, y, iters=1000, learning_rate=0.01)

new_data = np.array([165349.2, 136897.8, 471784.1, 'New York']).reshape(1, -1)  # Example new data
new_data_scaled = scaler.transform(ct.transform(new_data))

new_prediction = np.dot(np.append(1, new_data_scaled), theta)

print(f"Predicted value: {new_prediction[0]}")
data.head()

```

## Output:
# X&Y values:
![image](https://github.com/PYNAMVINODH/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742678/3d12f971-3ae5-4d8e-b851-bdc59c06d564)

 ![image](https://github.com/PYNAMVINODH/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742678/2603d3be-4a97-41ac-9192-343a2d31e077)
 
# X-Scaled & Y-Scaled:

![image](https://github.com/PYNAMVINODH/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742678/f7b7a8c1-0071-411b-a7be-3c73dcfbcfaa)


![image](https://github.com/PYNAMVINODH/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742678/b8d746f3-dc1c-4e49-a105-c5e4591b5b54)

# Predicted value:
![image](https://github.com/PYNAMVINODH/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742678/387720e9-d6e7-457d-a034-188cd2fa9dbd)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
