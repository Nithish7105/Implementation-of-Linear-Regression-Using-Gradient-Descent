# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: NITHISH KUMAR.B
RegisterNumber:  212223040134
*/
```
```PY
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```


## Output:
### Data Information
![310188551-a7888ba2-5abe-4057-9560-07ba652a980c](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/4015031e-f796-4985-b91e-2bcad67262af)

<br>
<br>
<br>

### Value of X
![310188658-5fde3790-29dc-4ab8-b3ab-a2f431fe3a00](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/ab2f2037-d0d3-4828-ad78-8ca22db3770d)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


### Value of X1_Scaled
![310188895-c5566457-572a-4d4b-adec-af4145ab8d83](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/6a38641e-c102-473b-9bf2-a6afeaabcdc6)

### Predicted Value
![310189062-117a2466-ed22-4985-bd40-849e5ccc5edf](https://github.com/gauthamkrishna7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/141175025/ae2ae2a4-16a5-4639-814d-5530c01c616d)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
