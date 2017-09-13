#coding:utf-8
#支持向量机模型的使用

from sklearn.svm import LinearSVC
import mglearn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

#例一，用LinearSVC模型，通过人工的增加一个feature来实现二分类
# X,y = make_blobs(centers=4, random_state=8)
# y = y%2

# # mglearn.discrete_scatter(X[:,0],X[:,1],y) #展示数据分布
# # plt.xlabel('Feature 0')
# # plt.ylabel('Feature 1')
# # plt.show()

# linear_svm = LinearSVC().fit(X,y)

# # mglearn.plots.plot_2d_separator(linear_svm,X) #展示拟合直线，显示出二维直线对其分类效果很差
# # mglearn.discrete_scatter(X[:,0],X[:,1],y)
# # plt.xlabel('Feature 0')
# # plt.ylabel('Feature 1')
# # plt.show()

# #再加入一个新feature，(feature1)^2，作为第三个维度，通过升维，在高维度上分类！！！
# X_new = np.hstack([X,X[:,1:]**2])# hstack([a,b]) 将几个列矩阵水平拼接，注意行数需要一致，用中括号括住，若写b[:,1:]报错

# from mpl_toolkits.mplot3d import Axes3D,axes3d
# figure = plt.figure()
# ax = Axes3D(figure,elev=-152,azim=-26)  ##创建三维坐标
# mask = y==0                            ##通过 mask 将y中值为零的项位置标出！！
# ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)  ##!!! 布尔取值！！！
# ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker='^',cmap=mglearn.cm2,s=60)  #因为正好是二分类问题，所以可以使用取反的方式取值
# ax.set_xlabel('Feature 0')
# ax.set_ylabel('Feature 1')
# ax.set_zlabel('Feature 1**2')

# # plt.show()
# #被放大后的数据在三维空间上很容易分离，再次使用线性模型分类
# linear_svm_3d = LinearSVC().fit(X_new,y)
# coef,intercept = linear_svm_3d.coef_.ravel(),linear_svm_3d.intercept_ #ravel()的用处就是将矩阵展平，N-D 变为 1-D
# xx = np.linspace(X_new[:,0].min()-2,X_new[:,0].max()+2,50)
# yy = np.linspace(X_new[:,1].min()-2,X_new[:,1].max()+2,50)

# XX,YY = np.meshgrid(xx,yy)#组成坐标点，(xx,yy)
# ZZ = (coef[0]*XX + coef[1]*YY +intercept)/-coef[2]
# ax.plot_surface(XX,YY,ZZ,rstride=8,cstride=8,alpha=0.3) #画出分界平面！！
# # ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
# # ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker='^',cmap=mglearn.cm2,s=60)
# # ax.set_xlabel('Feature 0')
# # ax.set_ylabel('Feature 1')
# # ax.set_zlabel('Feature 1**2')
# plt.show()

# #例二，用SVC模型，通过寻找support vector 来分类
# from sklearn.svm import SVC 
# X,y = mglearn.tools.make_handcrafted_dataset()
# svm = SVC(kernel='rbf',C=10,gamma=0.01).fit(X,y) 
# #gamma 设置gaussian kernel的半径，即gamma越小，半径越大，算法认为各个数据点距离越近。gamma越小所建模型越简单，分界线越平滑
# #C值设置各个数据点的影响大小，当C值很小的时候各个数据点的影响力很小，分界线接近直线；当C值增大时，各个数据点影响增大，分界线开始弯曲
# mglearn.plots.plot_2d_separator(svm,X,eps=0.5)
# mglearn.discrete_scatter(X[:,0],X[:,1],y)

# sv = svm.support_vectors_
# sv_labels = svm.dual_coef_.ravel()>0
# mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=10,markeredgewidth=3)# s设置数据点的大小值
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.show()

# #观察C和gamma两个参数对支持向量机模型的影响
# fig,axes = plt.subplots(3,3,figsize=(15,10))
# for ax,C in zip(axes,[-1,0,3]):
# 	for a, gamma in zip(ax,range(-1,2)):
# 		mglearn.plots.plot_svm(log_C=C,log_gamma=gamma,ax=a)

# axes[0,0].legend(['Class 0','Class 1','sv class 0','sv class 1'],ncol=4,loc=(0.9,1.2))
# plt.show()

#例三，RBF(the radial basis function) kernel SVM用于breast cancer dataset C和gamma使用默认值C=1,gamma=1/n_features
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)

svc = SVC()
svc.fit(X_train,y_train)

print('Accuracy on training set:{:.2f}'.format(svc.score(X_train,y_train)))
print('Accuracy on test set:{:.2f}'.format(svc.score(X_test,y_test)))
#因为支持向量机对数据有一定的要求，所以分类效果很不好
# #在对数坐标上将各个特征的最大值最小值画出来，可以看出处理前的数据最大值最小值差别巨大！
# plt.plot(X_train.min(axis=0),'o',label='min')##！！axis=0 代表现在选中的是行！！x_train.min(axis=0)返回x_train中最小的一行
# plt.plot(X_train.max(axis=0),'^',label='max')##！！axis=1 代表现在选中的是列！！x_train.max(axis=0)返回x_train中最大的一行
# plt.legend(loc='best')
# plt.xlabel('Feature index')
# plt.ylabel('Feature magnitude')
# plt.yscale('log')##将y轴设置为对数坐标！！！
# plt.show()

#将所有特征的值转化到0~1之间
#先找出X_train中最小的一行，让X_train所有行减去这一行，然后从这个新矩阵中找出最大的一行，然后新矩阵每一行除这个最大的一行
min_on_training = X_train.min(axis=0)
range_on_training = (X_train-min_on_training).max(axis=0)
X_train_scaled = (X_train-min_on_training)/range_on_training
#转化test数据集
min_on_testing = X_test.min(axis=0)
range_on_testing = (X_test-min_on_testing).max(axis=0)
X_test_scaled = (X_test-min_on_testing)/range_on_testing

# print('Minimum for each feature\n{}'.format(X_test_scaled.min(axis=0)))
# print('maxmun for each feature\n{}'.format(X_test_scaled.max(axis=0)))

#使用规整过的数据进行SVC建模
svc = SVC(C=1000)
svc.fit(X_train_scaled,y_train)

print('Accuracy on training set:{:.3f}'.format(svc.score(X_train_scaled,y_train)))
print('Accuracy on test set:{:.3f}'.format(svc.score(X_test_scaled,y_test)))