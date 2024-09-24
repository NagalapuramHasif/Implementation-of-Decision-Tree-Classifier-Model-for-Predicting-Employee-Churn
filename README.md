# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KOPPALA NAVEEN
RegisterNumber:212223100023  
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:
## data.head()

![image](https://github.com/user-attachments/assets/7d065ea3-00d5-43e0-abf7-97ed210ef7d0)


## data.info()

![image](https://github.com/user-attachments/assets/c8d48195-a74d-48f4-80a9-5b6281b3d47f)


## data.isnull().sum()

![image](https://github.com/user-attachments/assets/55a4db1e-6d53-4e62-9a45-f4d158f4d15e)


## data value count

![image](https://github.com/user-attachments/assets/e89b8584-0831-4ade-b7c9-678f740ce170)


## data.head() for salary

![image](https://github.com/user-attachments/assets/bf9432a7-80f4-4dbd-a056-20b644f7b60a)


## x.head()

![image](https://github.com/user-attachments/assets/a726c636-263d-47b1-a39b-9f681148db1d)


## accuracy value

![image](https://github.com/user-attachments/assets/c9b06831-34e4-4db5-9c98-36de65c339e4)


## data prediction

![image](https://github.com/user-attachments/assets/3e31aaed-6b62-4444-a0a1-f6145a44bb1d)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
