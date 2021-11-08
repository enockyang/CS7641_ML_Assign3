from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import fetch_mldata
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def best_confusion(r):
    n = r.shape[0]
    inp_list = [i for i in range(n)]
    permutations = list(itertools.permutations(inp_list))
    max_sum = 0
    max_list = (i for i in range(n))
    for i in permutations:
        cur = np.sum(np.diagonal(r[:,i]))
        if cur > max_sum:
            max_sum = cur
            max_list = i
    return r[:,max_list], max_sum


## load data
iris_dataset = load_iris()
iris_cluster = iris_dataset['data']
## load data
mnist = fetch_mldata('MNIST original', data_home = './datasets')
X, y = mnist['data'], mnist['target'] 
print(X.shape)
print(y.shape)
X_s, X_t, y_s, y_t = train_test_split(X, y, test_size = 0.5, random_state = 0)

## task1
### iris
estimator = KMeans(n_clusters=3)
estimator.fit(iris_cluster)
label_pred = estimator.labels_
r = confusion_matrix(iris_dataset['target'],label_pred)
estimator1 = GaussianMixture(n_components=3)
estimator1.fit(iris_cluster)
label_pred1 = estimator1.predict(iris_cluster)

### MNIST
estimator = KMeans(n_clusters=10)
estimator.fit(X)
label_pred = estimator.labels_
r = confusion_matrix(y,label_pred)
a, b  = best_confusion(r)
print(a)
print(b)

estimator1 = GaussianMixture(n_components=10)
estimator1.fit(X_s)
label_pred1 = estimator1.predict(X_s)
r1 = confusion_matrix(y_s,label_pred1)
a, b  = best_confusion(r1)
print(a)
print(b)

## task2
pca = PCA(n_components ='mle')
iris_pca = pca.fit_transform(iris_cluster)
iris_pca[0:10]
pca.explained_variance_ratio_

ica = FastICA()
iris_ica = ica.fit_transform(iris_cluster)
iris_ica[0:10]

rca = GaussianRandomProjection(n_components = 3, random_state = 10)
iris_rca = rca.fit_transform(iris_cluster)
iris_rca[0:10]

X_new = SelectKBest(chi2,k=3).fit_transform(iris_dataset['data'],iris_dataset['target'])
X_new[0:10]

## task3

estimator = KMeans(n_clusters=3)
estimator.fit(iris_pca)
label_pred = estimator.labels_
r = confusion_matrix(iris_dataset['target'],label_pred)
prin(r[:,[1,2,0]])

estimator = KMeans(n_clusters=3)
estimator.fit(iris_ica)
label_pred = estimator.labels_
r = confusion_matrix(iris_dataset['target'],label_pred)
print(r[:,[1,0,2]])


estimator = KMeans(n_clusters=3)
estimator.fit(iris_rca)
label_pred = estimator.labels_
r = confusion_matrix(iris_dataset['target'],label_pred)
print(r)

estimator = KMeans(n_clusters=3)
estimator.fit(X_new)
label_pred = estimator.labels_
r = confusion_matrix(iris_dataset['target'],label_pred)
print(r[:,[1,2,0]])

estimator1 = GaussianMixture(n_components=3)
estimator1.fit(iris_pca)
label_pred1 = estimator1.predict(iris_pca)
r1 = confusion_matrix(iris_dataset['target'],label_pred1)
print(r1[:,[1,0,2]])

estimator1 = GaussianMixture(n_components=3)
estimator1.fit(iris_ica)
label_pred1 = estimator1.predict(iris_ica)
r1 = confusion_matrix(iris_dataset['target'],label_pred1)
print(r1[:,[0,2,1]])

estimator1 = GaussianMixture(n_components=3)
estimator1.fit(iris_rca)
label_pred1 = estimator1.predict(iris_rca)
r1 = confusion_matrix(iris_dataset['target'],label_pred1)
print(r1[:,[1,2,0]])

estimator1 = GaussianMixture(n_components=3)
estimator1.fit(X_new)
label_pred1 = estimator1.predict(X_new)
r1 = confusion_matrix(iris_dataset['target'],label_pred1)
print(r1[:,[1,2,0]])

## task 4
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size = 0.5, random_state = 0)
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 10000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

X_train, X_test, Y_train, Y_test = train_test_split(iris_pca, iris_dataset['target'], test_size = 0.5, random_state = 0)
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 10000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

X_train, X_test, Y_train, Y_test = train_test_split(iris_ica, iris_dataset['target'], test_size = 0.5, random_state = 0)
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 10000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

X_train, X_test, Y_train, Y_test = train_test_split(iris_rca, iris_dataset['target'], test_size = 0.5, random_state = 0)
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 10000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

X_train, X_test, Y_train, Y_test = train_test_split(X_new, iris_dataset['target'], test_size = 0.5, random_state = 0)
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 10000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))


## task 5
estimator = KMeans(n_clusters=3)
estimator.fit(iris_cluster)
label_pred = estimator.labels_

irirs_cluster1 = np.c_[iris_cluster,label_pred]

X_train, X_test, Y_train, Y_test = train_test_split(irirs_cluster1, iris_dataset['target'], test_size = 0.5, random_state = 0)
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 10000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

estimator1 = GaussianMixture(n_components=3)
estimator1.fit(iris_cluster)
label_pred1 = estimator1.predict(iris_cluster)

irirs_cluster2 = np.c_[iris_cluster,label_pred1]

X_train, X_test, Y_train, Y_test = train_test_split(irirs_cluster2, iris_dataset['target'], test_size = 0.5, random_state = 0)
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 10000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))





