#coding:utf-8
#随机树林 构建5个决策树，推测半月形分布数据
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

# #用RandomForestClassifier分类两半月形分布的数据
# X,y = make_moons(n_samples=100,noise=0.25,random_state=3)
# X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

# forest = RandomForestClassifier(n_estimators=5,random_state=2)   #n_estimators用于设置建几个决策树，实际中设置的很大，从而使边界变得很平滑
# forest.fit(X_train,y_train) 

# fig,axes= plt.subplots(2,3,figsize=(20,10))
# for i,(ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
# 	ax.set_title('tree{}'.format(i))
# 	mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)

# mglearn.plots.plot_2d_separator(forest,X_train,fill=True,ax=axes[-1,-1],alpha=0.4)
# axes[-1,-1].set_title('Random Forest')
# mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
# # plt.show()
# print('the score of the train:{}'.format(forest.score(X_train,y_train)))
# print('the score of the test:{}'.format(forest.score(X_test,y_test)))

#用GradientBoostingClassifier分类cancer
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)

gbrt = GradientBoostingClassifier(random_state=0,max_depth=1,learning_rate=0.01)##max_depth,learning_rate 过大容易发生过拟合
gbrt.fit(X_train,y_train)

print('Accuracy on training set:{:.3f}'.format(gbrt.score(X_train,y_train)))
print('Accuracy on test set:{:.3f}'.format(gbrt.score(X_test,y_test)))

