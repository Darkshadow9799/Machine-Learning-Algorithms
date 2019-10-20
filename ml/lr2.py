from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)
data=pd.read_csv('headbrain.csv')
data.head()
print(data.shape)
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
m=len(X)
#Use rank 1 matrix in scikit learn
X=X.reshape((m,1))
#Creating model
reg=LinearRegression()
#Fitting training set
reg=reg.fit(X,Y)
#Y prediction
Y_pred=reg.predict(X)
#Calculating R2 Score
r2_score=reg.score(X,Y)
print(r2_score)