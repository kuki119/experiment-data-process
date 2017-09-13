#coding: utf-8
#以iris flower数据 来比较多个模型，选出最优模型，做预测

#load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url,names=names)
#dimensions of the dataset  how many instances(rows) and how many attributes(columns)
print(dataset.shape)
#peek at the data itself  the first 20 rows of the data
print(dataset.head(20))
#statistical summary of all attributes  this includes the count,mean,the min and max values as well as some percentiles
print(dataset.describe()) #返回一个统计表，统计各个特征的总个数、均值等统计量
#breakdown of the data by the class variable
print(dataset.groupby('class').size()) #返回每一类中包含的个体数量

#data visualization  
#这里用到单变量图和多变量图；单变量图利于理解单个特征，多变量图便于理解多个特征之间关系
#univariate plots
#1,box and whisker plots  画盒线图 包含5条横线，分别代表 最小值 25% 50% 75%分位点 和最大值
# dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# plt.show()
# #2，histograms 柱状图 可用于判断该特征大致服从什么分布
# dataset.hist()
# plt.show()
# #multivariate plots
# #scatter plot matrix  将所有特征两两组合画图，观察两者之间关系  ？？为什么对角线上是柱状图
# scatter_matrix(dataset)
# plt.show()

#split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2
seed = 7
X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size=0.2,random_state=7)

#test harness
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))