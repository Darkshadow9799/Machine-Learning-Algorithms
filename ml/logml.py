import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
titanic_data=pd.read_csv("train.csv")
titanic_data.drop("Cabin",axis=1,inplace=True)
titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull())
plt.show()
sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
Pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)
X=titanic_data.drop('Survived',axis=1)
y=titanic_data["Survived"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
print(accuracy_score(y_test,predictions))
print("New Input")
age=int(input('Enter your age '))
sp=int(input('Enter Sibsp (0:For false and 1:For True) '))
pa=int(input('Enter Parch (0:For false and 1:For True) '))
fa=float(input('Enter the fare '))
gender=int(input('0:For Female and 1:For Male '))
q=int(input('Q : (0:For false and 1:For True)'))
s=int(input('S: (0:For false and 1:For True)'))
tw=int(input('Passenger Class 2(0:For false and 1:For True)'))
th=int(input('Passenger Class 3(0:For false and 1:For True)'))
ynew=logmodel.predict([[age,sp,pa,fa,gender,q,s,tw,th]])
if(ynew==1):
	print('Survived')
else:
	print('Not Survived')