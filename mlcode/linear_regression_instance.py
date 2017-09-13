#coding:utf-8

import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import numpy as np

#线性拟合的第一个例子
# X,y = mglearn.datasets.make_wave(n_samples=60)
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

# fig,ax = plt.subplots(1,1)
# ax.scatter(X_train,y_train)
# ax.set_xlabel('X_train')
# ax.set_ylabel('y_train/y_predict')

# lr = LinearRegression().fit(X_train,y_train)
# print(lr.score(X_test,y_test))
# y_pre = lr.predict(X_train)
# ax.plot(X_train,y_pre,'r',markersize=5)
# ax.legend(['prediction','datasets'],loc='best')
# # plt.show()

# coefficient = lr.coef_  ##!!! lr.coef_ 里存放lr拟合直线的系数！！
# intercept = lr.intercept_   ##!!! lr.intercept_ 里存放lr拟合直线的截距！！
# print (coefficient,intercept)

#线性拟合第二个例子，波士顿房价,使用线性回归，在多特征数据中，表现出过拟合，不好
# X,y = mglearn.datasets.load_extended_boston()


# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# lr = LinearRegression().fit(X_train,y_train)

# a = lr.score(X_train,y_train)
# b = lr.score(X_test,y_test)

# print (a,'\n',b)
# print ('training set score:{:.2f}'.format(lr.score(X_train,y_train)))
# print ('test set score:{:.2f}'.format(lr.score(X_test,y_test)))

#线性拟合第二个例子，波士顿房价,使用ridge模型,更具有普适价值的模型 regularizing正则化 调节模型的复杂度
from sklearn.linear_model import Ridge

# x_alpha = np.arange(0,0.5,0.01)
# score_train = []
# score_test = []
# for i in x_alpha:
# 	ridge = Ridge(alpha=i).fit(X_train,y_train)
# 	score_train.append(ridge.score(X_train,y_train))
# 	score_test.append(ridge.score(X_test,y_test))

# plt.plot(x_alpha,score_train,label='score train')
# plt.plot(x_alpha,score_test,label='score test')
# plt.legend()
# plt.show()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
ridge = Ridge(alpha=10).fit(X_train,y_train)

print('train set score:{:.2f}'.format(ridge.score(X_train,y_train)))
print('test set score:{:.2f}'.format(ridge.score(X_test,y_test)))

#线性拟合第三个例子，波士顿房价，使用lasso模型 regularizing:L1 regularization
# from sklearn.linear_model import Lasso

# lasso = Lasso().fit(X_train,y_train)
# print('Training set score:{:.2f}'.format(lasso.score(X_train,y_train)))
# print('Test set score:{:.2f}'.format(lasso.score(X_test,y_test)))
# print('Number of features used:{}'.format(np.sum(lasso.coef_!=0)))

# lasso001 = Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train) #alpha调节模型的复杂程度，max_iter调节最大循环次数（十万以上）！！！ 
# print('Training set score:{:.2f}'.format(lasso001.score(X_train,y_train)))
# print('Test set score:{:.2f}'.format(lasso001.score(X_test,y_test)))
# print('Number of features used:{}'.format(np.sum(lasso001.coef_!=0)))