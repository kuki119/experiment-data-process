#coding:utf-8
#朴素贝叶斯模型使用 与 决策树模型的使用

import numpy as np 
import matplotlib.pyplot as plt

# X = np.array([[0,1,0,1],[1,0,1,1],[0,0,0,1],[1,0,1,0]])
# y = np.array([0,1,0,1])

# counts = {}
# for label in np.unique(y):
# 	counts[label] = X[y==label].sum(axis=0)# axis=0时纵向求和，axis=1时横向求和

# print('Feature counts:\n{}'.format(counts))

#决策树的使用
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

# cancer = load_breast_cancer()
# X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)

# tree = DecisionTreeClassifier(max_depth=4,random_state=0)##通过max_depth设置决策树的层数，可以降低过拟合，从而实现更好的普适性
# tree.fit(X_train,y_train)

# print('Accuracy on training set:{:.3f}'.format(tree.score(X_train,y_train)))
# print('Accuracy on test set:{:.3f}'.format(tree.score(X_test,y_test)))

# from sklearn.tree import export_graphviz
# export_graphviz(tree,out_file='tree.dot',class_names=['malignant','benign'],feature_names=cancer.feature_names,impurity=False,filled=True)

# print('Feature importances:{}'.format(tree.feature_importances_))#feature_importances_来查看各个特征的重要程度
# print('Feature:',len(tree.feature_importances_)) #特征总数为30个
# print('the sum of the feature importances:',sum(tree.feature_importances_))#所有的特征重要性只和为 1，单个特征重要性为小数

# def plot_feature_importances_cancer(model):
# 	plt.figure(figsize=(10,5))
# 	n_features = cancer.data.shape[1]
# 	plt.barh(range(n_features),model.feature_importances_,align='center',alpha=0.5,color='b')#bar用于画垂直于x轴的条形图，barh用于画垂直于y轴的条形图
# 	plt.yticks(np.arange(n_features),cancer.feature_names)
# 	plt.xlabel('Feature importance')
# 	plt.ylabel('Feature')
# 	plt.show()

# plot_feature_importances_cancer(tree)

#决策树与线性回归模型的预测对比
import pandas as pd 
ram_prices = pd.read_csv('D:/ram_price.csv')

# plt.semilogy(ram_prices.date,ram_prices.price)
plt.xlabel('Year')
plt.ylabel('Price in $/Mbyte')

from sklearn.tree import DecisionTreeRegressor
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train.date[:,np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train,y_train)
linear_reg = LinearRegression().fit(X_train,y_train)

X_all = ram_prices.date[:,np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
# plt.semilogy(ram_prices.date,ram_prices.price,'.',label = 'Real data')
plt.legend()
plt.show()