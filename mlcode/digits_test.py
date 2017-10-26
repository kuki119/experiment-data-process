
# coding: utf-8

# In[29]:

from sklearn import datasets
digits = datasets.load_digits()


# In[33]:

import pandas as pd
digits1 = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)


# In[37]:

print(digits.key)


# In[38]:

dir(digits)


# In[39]:

print(digits.data)
print(digits.target)
print(digits.DESCR)


# In[40]:

print(digits.keys)


# In[41]:

digits.data.shape


# In[42]:

digits.target.shape


# In[43]:

digits.DESCR.shape


# In[44]:

digits.images.shape


# In[58]:

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,7,str(digits.target[i]))
plt.show()


# In[59]:

randomized_pca = RandomizedPCA(n_components=2)


# In[61]:

from sklearn.decomposition import RandomizedPCA
randomized_pca = RandomizedPCA(n_components=2)
reduced_data_rpca = randomized_pca.fit_transform(digits.data)


# In[62]:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(digits.data)


# In[63]:

reduced_data_rpca.shape


# In[64]:

reduced_data_pca.shape


# In[65]:

reduced_data_rpca


# In[66]:

reduced_data_pca


# In[78]:

colors = ['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:,0][digits.target == i] 
    y = reduced_data_rpca[:,1][digits.target == i]
    plt.scatter(x,y,c=colors[i])
plt.legend(digits.target_names,bbox_to_anchor=(1,1),loc=2,borderaxespad=0)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Scatter Plot')
plt.show()


# In[79]:

from sklearn.preprocessing import scale
data = scale(digits.data)


# In[80]:

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test,images_train,images_test = train_test_split(data,digits.target,digits.images,test_size=0.25,random_state=42)


# In[81]:

n_samples,n_features = X_train.shape


# In[82]:

n_samples


# In[83]:

n_features


# In[84]:

n_digits = len(np.unique(y_train))


# In[85]:

n_digits


# In[96]:

from sklearn import cluster
clf = cluster.KMeans(init = 'k-means++',n_clusters = 10,random_state=42)
clf.fit(X_train,y_train)


# In[97]:

fig = plt.figure(figsize=(8,3))
fig.suptitle('cluster center images',fontsize=14,fontweight='bold')
for i in range(10):
    ax = fig.add_subplot(2,5,1+i)
    ax.imshow(clf.cluster_centers_[i].reshape((8,8)),cmap=plt.cm.binary)
    plt.axis('off')
plt.show()


# In[98]:

y_pred = clf.predict(X_test)


# In[99]:

y_pred[:100]


# In[100]:

y_test[:100]


# In[94]:

clf.cluster_centers.shape


# In[95]:

clf.cluster_centers_.shape


# In[101]:

from sklearn import metrics
print(metrics.confusion_matrix(y_test,y_pred))


# In[102]:

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          %(clf.inertia_,
      homogeneity_score(y_test, y_pred),
      completeness_score(y_test, y_pred),
      v_measure_score(y_test, y_pred),
      adjusted_rand_score(y_test, y_pred),
      adjusted_mutual_info_score(y_test, y_pred),
      silhouette_score(X_test, y_pred, metric='euclidean')))


# In[103]:

# Import `train_test_split`
from sklearn.cross_validation import train_test_split

# Split the data into training and test sets 
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

# Import the `svm` model
from sklearn import svm

# Create the SVC model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the SVC model
svc_model.fit(X_train, y_train)


# In[104]:

# Split the `digits` data into two equal sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=0)

# Import GridSearchCV
from sklearn.grid_search import GridSearchCV

# Set the parameter candidates
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X_train, y_train)

# Print out the results 
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)


# In[ ]:



