from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('/Users/sherrysheng/Downloads/NTU-20191018-master/data/tmdb_5000_features.csv')

# popularity & vote_count
data_1 = data[['popularity','vote_count']]
print(data_1.shape)
print(data_1.head())

f1 = data_1['popularity'].values
f2 = data_1['vote_count'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 100, 0, 10000])
plt.scatter(f1, f2, c='black', s=2)
plt.show()

# Euclidean Distance Caculator
# def dist(a, b, ax=1):
#     return np.linalg.norm(a - b, axis=ax)

kmeans = KMeans(n_clusters=3, max_iter=10000)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)
# plt.scatter(f1, f2, c='#050505', s=7)
# plt.scatter(X[:, 0], X[:, 1], marker='*', s=200, c='g')
# plt.show()

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=2)
plt.scatter(C[:, 0], C[:, 1], color='black', s=2)
plt.show()

# vote_average & vote_count
data_2 = data[['vote_average','vote_count']]
print(data_2.shape)
print(data_2.head())

f1 = data_2['vote_average'].values
f2 = data_2['vote_count'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 10, 0, 10000])
plt.scatter(f1, f2, c='black', s=2)
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=10000)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=2)
plt.scatter(C[:, 0], C[:, 1], color='black', s=2)
plt.show()


# runtime & vote_average
data_3 = data[['runtime','vote_average']]
print(data_3.shape)
print(data_3.head())

f1 = data_3['runtime'].values
f2 = data_3['vote_average'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 200, 0, 10])
plt.scatter(f1, f2, c='black', s=2)
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=10000)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=2)
plt.scatter(C[:, 0], C[:, 1], color='black', s=2)
plt.show()


# cast_num & vote_average
data_2 = data[['cast_num','vote_average']]
print(data_2.shape)
print(data_2.head())

f1 = data_2['cast_num'].values
f2 = data_2['vote_average'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 200, 0, 10])
plt.scatter(f1, f2, c='black', s=2)
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=10000)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=2)
plt.scatter(C[:, 0], C[:, 1], color='black', s=2)
plt.show()