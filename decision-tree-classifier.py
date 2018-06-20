#http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

balance_data = pd.read_csv('balance-scale.data.txt', sep=',', header= None)

print("Dataset length:: {}".format(len(balance_data)))
print("Dataset shape:: {}".format(balance_data.shape))
print("Dataset top five lines::")
print("{}".format(balance_data.head()))

X = balance_data.values[:, 1:5]
Y = balance_data.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5) #can choose gini or entropy for criterion

clf_gini.fit(X_train, y_train)

# print(clf_gini)
# foo = clf_gini.predict([[4,4,3,3]])
# print(foo)

y_pred = clf_gini.predict(X_test)

print(y_pred)

print("Accuracy is {}".format(accuracy_score(y_test,y_pred)*100)) #compare gini and entropy accuracy for best fit
