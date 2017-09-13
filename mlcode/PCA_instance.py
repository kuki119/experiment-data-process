#coding:utf-8

import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
import numpy as np 
import mglearn

#用breast cancer 数据展示PCA的降维功能
# cancer = load_breast_cancer()

# #将每个特征 以条形图的形式展示出来，可以用于观察那些特征可以很好的区分不同的类别
# # fig,axes = plt.subplots(15,2,figsize=(10,20))
# # malignant = cancer.data[cancer.target == 0]
# # benign = cancer.data[cancer.target == 1]

# # ax = axes.ravel()

# # for i in range(30):
# # 	_,bins = np.histogram(cancer.data[:,i],bins=50)
# # 	ax[i].hist(malignant[:,i],bins=bins,color=mglearn.cm3(0),alpha=0.5)
# # 	ax[i].hist(benign[:,i],bins=bins,color=mglearn.cm3(2),alpha=0.5)
# # 	# ax[i].set_title(cancer.feature_names[i])
# # 	ax[i].set_yticks(())

# # ax[0].set_xlabel('Feature magnitude')
# # ax[0].set_ylabel('Frequency')
# # ax[0].legend(['malignant','benign'],loc='best')
# # fig.tight_layout()

# # plt.show()

# #用PCA捕捉主要特征，并且对数据降维，实现可视化
# #在用PCA之前，要先对数据进行归一化
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(cancer.data)
# X_scaled = scaler.transform(cancer.data)

# from sklearn.decomposition import PCA 
# #使用PCA，首先，实例化并且设定要降到的维数；其次，fit 找到要投影的方向；最后，transform 进行选择降维
# pca = PCA(n_components=2)
# pca.fit(X_scaled)

# X_pca = pca.transform(X_scaled)
# print('Original shape:{}'.format(str(X_scaled.shape)))
# print('Reduced shape:{}'.format(str(X_pca.shape)))

# #展示训练完的PCA中的元素, 用 pca.components_ 查看
# print('PCA component shape:{}'.format(pca.components_.shape))
# print('PCA components:{}'.format(pca.components_))
# #用热分布图 展示 各个特征的重要程度
# plt.matshow(pca.components_,cmap='viridis')
# plt.yticks([0,1],['First component','Second component'])
# plt.colorbar()
# plt.xticks(range(len(cancer.feature_names)),
# 	cancer.feature_names,rotation=60,ha='left')
# plt.xlabel('Feature')
# plt.ylabel('Principal components')

# plt.show()

# #此时 仅保留了两个特征量，用二维图显示数据
# plt.figure(figsize=(8,8))
# mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target,alpha=0.7)
# plt.legend(cancer.target_names,loc='best')
# plt.gca().set_aspect('equal')
# plt.xlabel('First principal component')
# plt.ylabel('Second principal component')

# plt.show()

# #尝试保留三个特征，画三维空间图像
# pca = PCA(n_components=3)
# pca.fit(X_scaled)

# X_pca = pca.transform(X_scaled)
# target = cancer.target
# print('Target shape:{}'.format(str(target.shape)))
# print('Reduced shape:{}'.format(str(X_pca.shape)))

# from mpl_toolkits.mplot3d import Axes3D 
# mask = target==1 
# figure = plt.figure()
# ax = Axes3D(figure)
# ax.scatter(X_pca[mask,0],X_pca[mask,1],X_pca[mask,2],c='b')
# ax.scatter(X_pca[~mask,0],X_pca[~mask,1],X_pca[~mask,2],c='r',marker='^')
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Feature 3')

# plt.show()

#用人脸数据 展示PCA的特征提取功能
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

fix,axes = plt.subplots(2,5,figsize=(15,8),
	subplot_kw={'xticks':(),'yticks':()})
for target,image,ax in zip(people.target,people.images,axes.ravel()):
	ax.imshow(image)
	ax.set_title(people.target_names[target])

