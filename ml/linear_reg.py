import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)
data=pd.read_csv('headbrain.csv')
data.head()
print(data.shape)
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
mean_x=np.mean(X)
mean_y=np.mean(Y)
n=len(X)
numer=0
demon=0
m=len(X)
for i in range(m):
		numer+=(X[i]-mean_x)*(Y[i]-mean_y)
		demon+=(X[i]-mean_x)**2
b1=numer/demon
b0=mean_y-(b1*mean_x)
b1,b0
ss_t=0
ss_r=0
for i in range(m):
		y_pred=b0+b1*X[i]
		ss_t+=(Y[i]-mean_y)**2
		ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)
