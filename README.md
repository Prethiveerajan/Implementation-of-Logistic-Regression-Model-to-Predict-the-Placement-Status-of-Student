# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
```
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values 
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Prethiveerajan P
RegisterNumber:  212221230079
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data.isnull().sum()
data1.head()
data.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print (data1)
x=data1.iloc[:,:-1]
y=data1["status"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(solver="liblinear")
c.fit(x_train,y_train)
y_pred=c.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
acur=accuracy_score(y_test,y_pred)
acur
from sklearn.metrics import confusion_matrix
con=confusion_matrix(y_test,y_pred)
print(con)
from sklearn.metrics import classification_report
class_report=classification_report(y_test,y_pred)
print(class_report)
print(c.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output for head:
![output](head.png)
## Prediction:
![output](predicted.png)
## accuracy:
![output](accuracy.png)
## confusion :
![output](confusion.png)
## classification report:
![output](class.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
