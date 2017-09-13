#coding:utf-8
#机械学习 之 模型评估和提升

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = load_iris()

# #使用 cross_val_score 进行数据分割和交叉验证所建模型
# logreg = LogisticRegression()

# scores = cross_val_score(logreg,iris.data,iris.target,cv=3)
# print('Cross-validation scores:{}'.format(scores))
# print('Cross-validation scores:{:.2f}'.format(scores.mean()))

#将数据集分成三部分，第一部分用于训练模型
#第二部分用于模型的参数选择（即模型第一次打分）
#第三部分用于最终模型评判（即模型的第二次打分）

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# X_trainval,X_test,y_trainval,y_test = train_test_split(iris.data,iris.target,random_state=0)

# X_train,X_valid,y_train,y_valid = train_test_split(X_trainval,y_trainval,random_state=1)
# # print('size of training set:{}  size of validation set:{}  size of test set:{}'.format(
# #   X_train.shape[0],X_valid.shape[0],X_test.shape[0]))

# best_score = 0

# #使用单一的 train_test_split函数所产生的训练集 测试集的划分
# for gamma in [0.001,0.01,0.1,1,10,100]:
#   for C in [0.001,0.01,0.1,1,10,100]:
#       svm = SVC(gamma=gamma,C=C)
#       svm.fit(X_train,y_train)

#       score = svm.score(X_valid,y_valid)

#       if score > best_score:
#           best_score = score
#           best_parameters = {'C':C,'gamma':gamma}

# svm = SVC(**best_parameters)
# svm.fit(X_trainval,y_trainval)
# test_score = svm.score(X_test,y_test)
# print('Best score on validation set:{:.2f}'.format(best_score))
# print('Best parameters:', best_parameters)
# print('Test set score with best parameters:{:.2f}'.format(test_score))            

# #使用 cross_val_score 函数，产生交叉验证，并用grid search 来找最优参数，效果很好！！0.97！！
# for gamma in [0.001,0.01,0.1,1,10,100]:
#   for C in [0.001,0.01,0.1,1,10,100]:
#       svm = SVC(gamma=gamma,C=C)
#       scores = cross_val_score(svm,X_trainval,y_trainval,cv=5)
#       score = np.mean(scores)
#       if score > best_score:
#           best_score = score
#           best_parameters = {'C':C,'gamma':gamma}

# svm = SVC(**best_parameters) #这里必须有两个 ** ，为什么，不清楚……
# svm.fit(X_trainval,y_trainval)

# test_score = svm.score(X_test,y_test)
# print('best score on validation set:{:.2f}'.format(best_score))
# print('best parameters:',best_parameters)
# print('test set score with best parameters:{:.2f}'.format(test_score))

#直接使用 GridSearchCV 类，来完成 grid search 和 cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

# param_grid = {'C':[0.001,0.01,0.1,1,10,100,200,300],'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10,100]}

# #GridSearchCV 函数，把想要的算法扔进去、要试验的参数集以字典的形式扔进去、指定几分训练集即可！！！
# grid_search = GridSearchCV(SVC(),param_grid,cv=5)

# X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=0)

# # grid_search对象类似于之前分类器，也有 fit \ predict \ score 方法
# grid_search.fit(X_train,y_train)

# print('test set score: {:.2f}'.format(grid_search.score(X_test,y_test)))
# print('best parameters:{}'.format(grid_search.best_params_))
# print('best cross_validation score:{:.2f}'.format(grid_search.best_score_))
# print('best estimator:\n{}'.format(grid_search.best_estimator_))

# results = pd.DataFrame(grid_search.cv_results_)
# # print(results.head())

# #需要学会 heat map的绘制！！用于查看自己的参数选取范围是否合理！！！
# scores = np.array(results.mean_test_score).reshape(-1,8)
mglearn.tools.heatmap(scores,xlabel='gamma',xticklabels=param_grid['gamma'],
    ylabel='C',yticklabels=param_grid['C'])

# plt.show()

#用 Pipeline 将数据的预处理 和 选取的模型 放在一起处理
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# pipe = Pipeline ([('scaler',MinMaxScaler()),('svm',SVC())])

# pipe.fit(X_train,y_train)

# print('test score:{:.2f}'.format(pipe.score(X_test,y_test)))

# #使用pipeline最大的好处是，在GridSearchCV中只需要使用一个 estimator， 避免了预处理一个，训练模型一个
# param_grid = {'svm__C':[0.001,0.01,0.1,1,10,100],'svm__gamma':[0.001,0.01,0.1,1,10,100]}
# #注意 这里的参数字典，必须指明是向那个estimator传递参数， estimator__param  !!!

# grid = GridSearchCV(pipe,param_grid=param_grid,cv=5)
# grid.fit(X_train,y_train)
# print('best cross-validation accuracy:{:.2f}'.format(grid.best_score_))
# print('test set score:{:.2f}'.format(grid.score(X_test,y_test)))
# print('best parameters:{}'.format(grid.best_params_))


#使用Grid-search 决定使用哪个模型（SVC 和 RandomForestClassifier）

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

pipe = Pipeline([('preprocessing',StandardScaler()),('classifier',SVC())])

param_grid = [
    {'classifier':[SVC()],'preprocessing':[StandardScaler(),None],
    'classifier__gamma':[0.001,0.01,0.1,1,10,100],
    'classifier__C':[0.001,0.01,0.1,1,10,100]},
    {'classifier':[RandomForestClassifier(n_estimators=100)],
    'preprocessing':[None],'classifier__max_features':[1,2,3]}]

X_train,X_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,random_state=0)

grid = GridSearchCV(pipe,param_grid,cv=5)
grid.fit(X_train,y_train)

print('best params:\n{}\n'.format(grid.best_params_))
print('best cross-validation score:{:.2f}'.format(grid.best_score_))
print('test-set score:{:.2f}'.format(grid.score(X_test,y_test)))
