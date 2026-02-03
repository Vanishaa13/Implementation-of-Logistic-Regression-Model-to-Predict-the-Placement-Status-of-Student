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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())


data = data.drop(["sl_no", "salary"], axis=1)



data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})


X = data.drop("status", axis=1)
y = data["status"]


X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = model.decision_function(X_test)


z_sorted = np.sort(z)
sigmoid_values = sigmoid(z_sorted)


plt.figure(figsize=(8, 5))
plt.plot(z_sorted, sigmoid_values)
plt.xlabel("Linear Model Output (z)")
plt.ylabel("Predicted Probability")
plt.title("Sigmoid Curve - Logistic Regression")
plt.grid(True)
plt.show() 
*/
```

## Output:
<img width="561" height="782" alt="Screenshot 2026-01-31 143909" src="https://github.com/user-attachments/assets/e23083c4-5c85-4545-801a-5e44e03109cc" />

<img width="570" height="436" alt="Screenshot 2026-01-31 143918" src="https://github.com/user-attachments/assets/c80df219-8fff-42f5-9b59-29a7404d19db" />

<img width="942" height="614" alt="Screenshot 2026-01-31 144439" src="https://github.com/user-attachments/assets/9526c82c-d894-4d58-8efa-b05950a42bb2" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
