#coding:utf-8
#knn K-邻近分类算法

# #第一个例子，二分类问题，首先熟悉K邻近算法的使用过程
# import mglearn
# X,y = mglearn.datasets.make_forge() #导入数据集
# from sklearn.neighbors import KNeighborsClassifier
# #数据可视化
# import matplotlib.pyplot as plt
# fig,axes = plt.subplots(1,3,figsize=(15,5)) #产生了3个图，fig放图像，axes放轴

# #画出选择1,3,9个临近点时的分解情况图
# for n_neighbors , ax in zip([1,3,9],axes):
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
#     mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=0.3)
#     #fill表示填充色是否开启；eps表示边距大小，即eps越大，数据区域越小；alpha表示分界线颜色深浅，0~1
#     mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
#     ax.set_title('{} neighbor(s)'.format(n_neighbors))
#     ax.set_xlabel('feature 0')
#     ax.set_ylabel('feature 1')
# axes[0].legend(loc=3)
# plt.show()
# print(type(fig))

# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0) #对数据划分为训练集和验证集

# # print(X_train.shape)


# clf = KNeighborsClassifier(n_neighbors = 3)
# clf.fit(X_train,y_train)

# # print('y_test:{}'.format(y_test))
# print('Test set predictions:{}'.format(clf.predict(X_test)))
# print('Test set accuracy:{:.2f}'.format(clf.score(X_test,y_test)))


#第二个例子，癌症的诊断分类,展示k值大小分别对训练准确度和测试准确度大小的影响
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,
    stratify=cancer.target,random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,19)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
    #训练集不能与测试集重合，否则准确率会出现跟随的状态！！！
    # clf.fit(X,y)
    # training_accuracy.append(clf.score(X,y))
    # test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings,training_accuracy,label='training accuracy')
plt.plot(neighbors_settings,test_accuracy,label='testing accuracy')
plt.xlabel('Accuracy')
plt.ylabel('n_neighbors')
plt.legend()
plt.show()