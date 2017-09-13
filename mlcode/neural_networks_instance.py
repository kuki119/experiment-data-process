#coding:utf-8
#使用neural networks (deeplearning)模型实例：

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
import mglearn

# X,y = make_moons(n_samples=100,noise=0.25,random_state=3)
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

# mlp = MLPClassifier(solver='lbfgs',activation='tanh',alpha=0.001,random_state=0,hidden_layer_sizes=[100,100]).fit(X_train, y_train)
# #activation 用来选择中间层使用那种非线性函数reul or tangens hyper
# mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.3)
# mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.show()

# print('Accuracy of the train set:{:.3f}'.format(mlp.score(X_train,y_train)))
# print('Accuracy of the test set:{:.3f}'.format(mlp.score(X_test,y_test)))
# plt.show()

# #循环改变隐藏的层数和节点和alpha值（罚值），观察各个参数对模型拟合情况的影响
# fig,axes = plt.subplots(2,4,figsize=(20,8))
# for axx,n_hidden_nodes in zip(axes,[10,100]):
# 	for ax,alpha in zip(axx,[0.0001,0.01,0.1,1]):
# 		mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes],alpha=alpha)
# 		mlp.fit(X_train,y_train)
# 		mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.3,ax=ax)
# 		mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
# 		ax.set_title('n_hidden=[{},{}]\nalpha={:,.4f}'.format(n_hidden_nodes,n_hidden_nodes,alpha))
# plt.show()

# #循环改变random_state的值，观察相同参数下的拟合结果:不同的random_state最后得到的模型也不相同
# fig,axes = plt.subplots(2,4,figsize=(20,8))
# for i,ax in enumerate(axes.ravel()):
# 	mlp = MLPClassifier(solver='lbfgs',random_state=i,hidden_layer_sizes=[100,100])
# 	mlp.fit(X_train,y_train)
# 	mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=0.3,ax=ax)
# 	mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
# 	ax.set_title('random_state={}'.format(i))
# plt.show()

#使用神经网络处理breast cancer数据
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# print('cancer data per-feature maxima:{}'.format(cancer.data.max(axis=0)))
# print('cancer data per-feature minima:{}'.format(cancer.data.min(axis=0)))

#直接进行模型训练
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)

# mlp = MLPClassifier(random_state=42)
# mlp.fit(X_train,y_train)

# print('Accuracy on training set:{:.2f}'.format(mlp.score(X_train,y_train)))
# print('Accuracy on test set:{:.2f}'.format(mlp.score(X_test,y_test)))

#先进行数据归一化，然后进行模型训练
# #归一化的方法一：
# min_on_X = cancer.data.min(axis=0)
# range_on_X = (cancer.data - min_on_X).max(axis=0)
# scale_on_X = (cancer.data - min_on_X)/range_on_X
#归一化方法二：
# mean_on_data = cancer.data.mean(axis=0)
# std_on_data = cancer.data.std(axis=0)
# scale_on_X = (cancer.data - mean_on_data)/std_on_data
#使用减去均值(mean)除以标准差(standard deviation),最终数据变成，mean=0,std=1

mean_on_data = X_train.mean(axis=0)
std_on_data = X_train.std(axis=0)
X_train_scaled = (X_train - mean_on_data)/std_on_data
X_test_scaled = (X_test - mean_on_data)/std_on_data

print('cancer data post-feature maxima:{}'.format(X_train_scaled.max(axis=0)))
print('cancer data post-feature minima:{}'.format(X_train_scaled.min(axis=0)))

# X_train,X_test,y_train,y_test = train_test_split(scale_on_X,cancer.target,random_state=0)

mlp = MLPClassifier(max_iter=1000,alpha=1,random_state=0)
mlp.fit(X_train_scaled,y_train)

print('Accuracy on training set:{:.3f}'.format(mlp.score(X_train_scaled,y_train)))
print('Accuracy on test set:{:.3f}'.format(mlp.score(X_test_scaled,y_test)))

#绘制出各个feature的重要程度：
plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0],interpolation='none',cmap='viridis')
plt.yticks(range(30),cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()
plt.show()

