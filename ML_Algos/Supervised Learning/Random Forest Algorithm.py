import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1)
clf = RandomForestClassifier(min_samples_split= 10, random_state= 4)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(20, 20))
export_graphviz(clf.estimators_[8],
                out_file='tree.dot',
                feature_names = data.feature_names,
                class_names = data.target_names,
                rounded = True,
                proportion = False,
                precision = 2,
                filled = True)
plt.show()
print(clf.estimators_)