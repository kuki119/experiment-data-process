#coding:utf-8

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

#用线性模型来分类，使用支持向量机SVM和LogisticRegression（分类模型）,注意LogisticRegression 与 LinearRegression区别
#LinearSVC 与 LogisticRegression都是靠参数c来调节正则化强弱的
# X,y = mglearn.datasets.make_forge()
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

# fig,axes = plt.subplots(1,2,figsize=(10,3))

# for model, ax in zip([LinearSVC(C=100),LogisticRegression(C=100)],axes):#zip([LinearSVC(),LogisticRegression()])相当于假设空间即模型空间？？
# 	#C越大，train_set拟合越好，但是预测能力可能会降低
# 	clf = model.fit(X_train,y_train)
# 	mglearn.plots.plot_2d_separator(clf,X,fill=False,eps=0.5,ax=ax,alpha=.7)#这条语句用于画分界线decision boundary（分界平面或分界超平面）
# 	mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)#这条语句用于画出各个数据点
# 	ax.set_title('{}'.format(clf.__class__.__name__))
# 	ax.set_xlabel('Feature 0')
# 	ax.set_ylabel('Feature 1')
# 	print('the score of {}:{:.2f}'.format(clf.__class__.__name__,clf.score(X_test,y_test)))

# axes[0].legend(loc='best')
# plt.show()

#用LinearLogistic 分类breast cancerdataset

# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)

# logreg = LogisticRegression(C=0.01).fit(X_train,y_train)
# print('Training set score:{:.3f}'.format(logreg.score(X_train,y_train)))
# print('Test set score:{:.3f}'.format(logreg.score(X_test,y_test)))

#用LinearSVC分类多个类别对象
from sklearn.datasets import make_blobs

X,y = make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.legend(['Class 0','Class 1','Class 2'])

linear_svm = LinearSVC().fit(X,y)
print('Coefficient shape:',linear_svm.coef_.shape)
print('intercept shape:',linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
	plt.plot(line,-(line*coef[0] + intercept) / coef[1],c=color) # w1*x1+w2*x2+b=0; 所以 第二个特征 x2=-(w1*x1+b)/w2

plt.ylim(-10,15)
plt.xlim(-10,8)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0','Class 1','Class 2','Line class 0','Line class 1','Line class 2'],loc='best')
plt.show()
