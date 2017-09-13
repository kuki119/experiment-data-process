#coding:utf-8
#对分类器的不确定性评估：decision_function and predict_proba

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs,make_circles
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

# X,y = make_circles(noise=0.25,factor=0.5,random_state=1)
# y_named = np.array(['blue','red'])[y] #只有当y是0或者1组成的情况下才能用blue替换原来的0，red替换原来的1
# # print(y.shape)
# # print(y_named.shape)

# X_train,X_test,y_train_named,y_test_named = train_test_split(X,y_named,random_state=0) 
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0) 

# gbrt = GradientBoostingClassifier(random_state=0)
# gbrt.fit(X_train,y_train_named)

# # print('Accuracy on training set:{:.2f}'.format(gbrt.score(X_train,y_train_named)))
# # print('Accuracy on test set:{:.2f}'.format(gbrt.score(X_test,y_test_named)))

# # print('X_test.shape:{}'.format(X_test.shape))
# # print('Decision function shape:{}'.format(gbrt.decision_function(X_test).shape))
# # print('Decision function:\n{}'.format(gbrt.decision_function(X_test)[:6]))

# greater_zero = (gbrt.decision_function(X_test)>0).astype(int)
# pred = gbrt.classes_[greater_zero]
# print('pred is equal to predictions:{}'.format(np.all(pred == gbrt.predict(X_test))))
# print('the elements of gbrt.classes_:',gbrt.classes_)

# #用于展示 decision_function
# fig,axes = plt.subplots(1,2,figsize=(13,5))
# mglearn.tools.plot_2d_separator(gbrt,X,ax=axes[0],alpha=0.4,fill=True,cm=mglearn.cm2)
# #显示二维平面上所有点被分到特定类别的强烈程度，用颜色深浅表征
# scores_image = mglearn.tools.plot_2d_scores(gbrt,X,ax=axes[1],alpha=0.4,cm=mglearn.ReBl)

# for ax in axes:
# 	mglearn.discrete_scatter(X_test[:,0],X_test[:,1],y_test,markers='^',ax=ax)
# 	mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,markers='o',ax=ax)
# 	ax.set_xlabel('Feature 0')
# 	ax.set_ylabel('Feature 1')

# # cbar = plt.colorbar(scores_image,ax=axes.tolist())
# axes[0].legend(['Test class 0','Test class 1','Train class 0','Train class 1'],ncol=4,loc=(0.1,1.1))
# plt.show()

#用于展示 predicting probabilities
# print('Predicted probabilities:\n{}'.format(gbrt.predict_proba(X_test[:6])))
# fig,axes = plt.subplots(1,2,figsize=(13,5))

# mglearn.tools.plot_2d_separator(gbrt,X,ax=axes[0],alpha=0.4,fill=True,cm=mglearn.cm2)
# scores_image = mglearn.tools.plot_2d_scores(gbrt,X,ax=axes[1],alpha=0.5,cm=mglearn.ReBl,function='predict_proba')

# for ax in axes:
# 	mglearn.discrete_scatter(X_test[:,0],X_test[:,1],y_test,markers='^',ax=ax)
# 	mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,markers='o',ax=ax)
# 	ax.set_xlabel('Feature 0')
# 	ax.set_ylabel('Feature 1')

# cbar = plt.colorbar(scores_image,ax=axes.tolist())
# axes[0].legend(['Test class 0','Test class 1','Train class 0','Train class 1'],ncol=4,loc=(0.1,1.1))
# plt.show()

##不确定性检验用于多类别分类问题
from sklearn.datasets import load_iris

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01,random_state=0)
gbrt.fit(X_train,y_train)

print('Decision function shape:{}'.format(gbrt.decision_function(X_test).shape))
print('Decision function:\n{}'.format(gbrt.decision_function(X_test)[:6,:]))

print('Argmax of decision function:\n{}'.format(np.argmax(gbrt.decision_function(X_test),axis=1)))
print('Predictions:\n{}'.format(gbrt.predict(X_test)))

print('Predicted probabilities:\n{}'.format(gbrt.predict_proba(X_test)[:6]))
print('sums:{}'.format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))

print('Argmax of predicted probabilities:\n{}'.format(
	np.argmax(gbrt.predict_proba(X_test),axis=1)))
print('Predictions:\n{}'.format(gbrt.predict(X_test)))