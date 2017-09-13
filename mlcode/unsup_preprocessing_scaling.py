#coding:utf-8
#简单的数据放缩用于数据的预处理，从而使后续的监督学习的精度提高
#具体方法有四种：StandardScaler,RobustScaler,MinMaxScaler,Normalizer
import matplotlib.pyplot as plt 
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# mglearn.plots.plot_scaling()
# plt.show()

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)

#用SVC模型拟合cancer原始数据，看拟合效果
from sklearn.svm import SVC
svm = SVC(C=100)
svm.fit(X_train,y_train)
print('Training set accuracy:{:.2f}'.format(svm.score(X_train,y_train)))
print('Test set accuracy:{:.2f}'.format(svm.score(X_test,y_test)))

#使用MinMaxScaler进行数据预处理
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# print('transformed shape:{}'.format(X_train_scaled.shape))
# print('per-feature minimum before scaling:\n{}'.format(X_train.min(axis=0)))
# print('per-feature maximum before scaling:\n{}'.format(X_train.max(axis=0)))
# print('per-feature minimum after scaling:\n{}'.format(X_train_scaled.min(axis=0)))
# print('per-feature maximum after scaling:\n{}'.format(X_train_scaled.max(axis=0)))

X_test_scaled = scaler.transform(X_test)
# print('per-feature minimum after scaling:\n{}'.format(X_test_scaled.min))

svm.fit(X_train_scaled,y_train)
print('Scaled training set accuracy:{:.2f}'.format(svm.score(X_train_scaled,y_train)))
print('Scaled test set accuracy:{:.2f}'.format(svm.score(X_test_scaled,y_test)))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled_std = scaler.transform(X_train)
X_test_scaled_std = scaler.transform(X_test)

svm.fit(X_train_scaled_std,y_train) 
print('SVM train accuracy:{:.2f}'.format(svm.score(X_train_scaled_std,y_train)))
print('SVM test accuracy:{:.2f}'.format(svm.score(X_test_scaled_std,y_test)))



# #用于监督学习的训练数据(X_train)和测试数据(X_test)必须使用相同的transformation
# #下面说明不使用相同的transformation 有什么
# from sklearn.datasets import make_blobs
# X,_ = make_blobs(n_samples=50,centers=5,random_state=4,cluster_std=2)
# X_train,X_test = train_test_split(X,random_state=5,test_size=0.1)

# fig,axes = plt.subplots(1,3,figsize=(13,4))

# #展示原始数据
# axes[0].scatter(X_train[:,0],X_train[:,1],c=mglearn.cm2(0),label='Training set',s=60)
# axes[0].scatter(X_test[:,0],X_test[:,1],marker='^',c=mglearn.cm2(1),label='Test set',s=60)
# axes[0].legend(loc='upper left')
# axes[0].set_title('Original Data')

# #展示正确放缩后的数据
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # print('the minimum of X_train:\n{}'.format(X_train_scaled.min(axis=0)))
# # print('the maximum of X_train:\n{}'.format(X_train_scaled.max(axis=0)))
# # print('the minimum of X_test:\n{}'.format(X_test_scaled.min(axis=0)))
# # print('the maximum of X_test:\n{}'.format(X_test_scaled.max(axis=0)))

# axes[1].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label='Training set',s=60)
# axes[1].scatter(X_test_scaled[:,0],X_test_scaled[:,1],marker='^',c=mglearn.cm2(1),label='Test set',s=60)
# axes[1].set_title('Scaled Data')

# #展示错误放缩后的数据 若X_train 与 X_test使用不同的transform,产生的结果就是两者放缩不一致
# test_scaler = MinMaxScaler()
# test_scaler.fit(X_test)
# X_test_scaled_bad = test_scaler.transform(X_test)

# axes[2].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label='Training set',s=60)
# axes[2].scatter(X_test_scaled_bad[:,0],X_test_scaled_bad[:,1],marker='^',c=mglearn.cm2(1),label='Test bad set',s=60)
# axes[2].set_title('Improperly Scaled Data')

# for ax in axes:
# 	ax.set_xlabel('Feature 0')
# 	ax.set_ylabel('Feature 1')

# plt.show()