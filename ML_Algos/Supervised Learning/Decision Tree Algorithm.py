'''
ID3 uses Information Gain : One with max Info Gain is selected.
    Info Gain(T, X) = Entropy(T) - Entropy(T, X)

C4.5 uses Gain ratio : One with max Gain ratio is selected.
    Split_Info(T, X) = -summation(P(c) logP(c))  {Log to the base 2}
    Gain Ratio(T, X) = Info Gain(T, X) / Split_Info(T, X)

CART uses Gini Index : One with lowest Gini Index value will be choosen.
    Gini Index = 1 - summation(Pi)^2
    where Pi is the probability of an object being classified to a particular class.

'''

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled= True)
plt.show()

