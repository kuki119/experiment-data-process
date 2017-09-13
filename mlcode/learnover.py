#coding:utf-8
#重新学习 决策树 随机森林 支持向量机，配合交叉验证

#1、决策树 + 交叉验证  决策树使用 max_depth 参数来限制树的深度，预砍伐，来保证模型的普适性
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mglearn

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)


# param_grid = {'max_depth':[2,3,4,5,6]}
# grid = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=10)
# grid.fit(X_train,y_train)
# tree = DecisionTreeClassifier(max_depth=3,random_state=0)
# tree.fit(X_train,y_train)
# # scores = cross_val_score(tree,cancer.data,cancer.target,cv=10)

# # print('cross_validation score:{}'.format(scores))
# # print('the mean score of cross_validation:{}'.format(scores.mean()))
# print('grid_search score:',grid.score(X_test,y_test))
# print('the best parameter:{}'.format(grid.best_params_))
# # print('test-set score:{:.2f}'.format(grid.score(X_test,y_test)))

# #查看各个特征的重要性：需要用训练过后的tree
# print('tree score:{:.2f}'.format(tree.score(X_test,y_test)))
# print('feature importances:',tree.feature_importances_)

#ensembles of decision trees：random forests ** gradient boost decision trees
#随机森林用于 cancer 数据集
from sklearn.ensemble import RandomForestClassifier

# forest = RandomForestClassifier(n_estimators=100,random_state=0)
# #可以用 max_depth 和 max_features 调节模型，max_depth -- 预砍；max_features -- 选几个特征
# forest.fit(X_train,y_train)

# print('accuracy on training set:{:.3f}'.format(forest.score(X_train,y_train)))
# print('accuracy on test set:{:.3f}'.format(forest.score(X_test,y_test)))
# print('importance of each features:\n{}'.format(forest.feature_importances_))
# print(forest)
# #用 feature_importances_ 来查看各个特征的重要性，各个特征重要性之和 = 1

# #使用数据集的交叉验证
# scores = cross_val_score(forest,cancer.data,cancer.target,cv=10)
# print('cross-validation scores:\n{}'.format(scores))
# print('mean score:{}'.format(scores.mean()))

# #使用 GridSearchCV 找最佳参数
# #对于分类问题，max_features=sqrt(n_features)  对于回归问题，max_features=log2(n_features)
# grid_params = {'max_depth':[3,6,10],'max_features':[5,15,25]}

# grid_search = GridSearchCV(RandomForestClassifier(n_estimators=100),grid_params,cv=5)
# grid_search.fit(X_train,y_train)

# print('the score on test set:{:.3f}'.format(grid_search.score(X_test,y_test)))
# print('the best score on CV set:{:.3f}'.format(grid_search.best_score_))
# print('the best estimator:{}'.format(grid_search.best_estimator_))
# print('the best parameters:{}'.format(grid_search.best_params_))

# #画测试图的 热分布图
# results = pd.DataFrame(grid_search.cv_results_)
# grid_scores = np.array(results.mean_test_score).reshape(-1,3)
# print(grid_scores)

# mglearn.tools.heatmap(grid_scores,xlabel='max_depth',xticklabels=grid_params['max_depth'],
#     ylabel='max_features',yticklabels=grid_params['max_features'])

# plt.show()

# #使用随机森林回归算法：
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import load_boston

# boston = load_boston()

# X_train,X_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=0)

# forest_reg = RandomForestRegressor(n_estimators=10000,n_jobs=-1)
# scores = cross_val_score(forest_reg,X_train,y_train,cv=10)

# print('the score on train set:{}'.format(scores))
# print('the mean score on train set:{}'.format(scores.mean()))

#使用支持向量机：
