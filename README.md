# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import required libraries.

2.Load the student placement dataset.

3.Select CGPA and internship status as input features.

4.Select placement status as the output variable.

5.Create the Logistic Regression model.

6.Train the model using the training data.

7.Predict the placement status.

8.Plot the graph showing placement results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Vanishaa harshini.B.R
RegisterNumber:212225040481
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    'CGPA': [6.5, 7.0, 8.2, 5.8, 9.1, 6.0, 8.5, 7.8],
    'Internship': [0, 1, 1, 0, 1, 0, 1, 1],
    'Placement': [0, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['CGPA', 'Internship']]
y = df['Placement']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Scatter plot
plt.figure()
for value in [0, 1]:
    subset = df[df['Placement'] == value]
    plt.scatter(subset['CGPA'], subset['Internship'])

# Decision boundary
x_values = np.linspace(df['CGPA'].min(), df['CGPA'].max(), 100)
y_values = -(model.coef_[0][0] * x_values + model.intercept_[0]) / model.coef_[0][1]

plt.plot(x_values, y_values)
plt.xlabel("CGPA")
plt.ylabel("Internship")
plt.title("Logistic Regression – Placement Status")
plt.show() 
*/
```

## Output:
<img width="941" height="582" alt="Screenshot 2026-01-30 143317" src="https://github.com/user-attachments/assets/fb3e3a19-eb24-4f7d-ada3-775dcbe7f0d0" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
