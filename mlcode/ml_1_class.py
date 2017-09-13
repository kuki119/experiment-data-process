# coding:utf-8

import numpy as np 
import sklearn  
import pandas as pd

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# # create dataframe from data in X_train
# # label the columns using the strings in iris_dataset.feature_names
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# # create a scatter matrix from the dataframe, color by y_train
# grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
# hist_kwds={'bins': 20}, s=60, alpha=.8, cmap='mglearn.cm3')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) #这里的neighbors值应该怎么选？？

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc1 = np.mean(y_pred == y_test)

acc2 = knn.score(X_test,y_test)

print (acc2)