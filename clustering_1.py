import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('seaborn-whitegrid')

# Importing the dataset
data = pd.read_csv('/path/to/data/tmdb_5000_features.csv')

# popularity & vote_count
data_1 = data[['popularity','vote_count']]
print(data_1.shape)
print(data_1.head())

f1 = data_1['popularity'].values
f2 = data_1['vote_count'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 100, 0, 10000])
plt.xlabel("Popularity")
plt.ylabel("Vote Count")
plt.scatter(f1, f2, c='black', s=3)
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=100)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=3)
plt.scatter(C[:, 0], C[:, 1], color='black', s=3)
plt.show()


# vote_average & vote_count
data_2 = data[['vote_average','vote_count']]
print(data_2.shape)
print(data_2.head())

f1 = data_2['vote_average'].values
f2 = data_2['vote_count'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 10, 0, 10000])
plt.xlabel("Vote Average")
plt.ylabel("Vote Count")
plt.scatter(f1, f2, c='black', s=3)
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=100)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=3)
plt.scatter(C[:, 0], C[:, 1], color='black', s=3)
plt.show()


# runtime & vote_average
data_3 = data[['runtime','vote_average']]
print(data_3.shape)
print(data_3.head())

f1 = data_3['runtime'].values
f2 = data_3['vote_average'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 200, 0, 10])
plt.xlabel("Runtime")
plt.ylabel("Vote Average")
plt.scatter(f1, f2, c='black', s=3)
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=100)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=3)
plt.scatter(C[:, 0], C[:, 1], color='black', s=3)
plt.show()


# Importing the dataset
data = pd.read_csv('/path/to/data/genre.csv')

# genre & vote_average
df1 = data[['genre_0','vote_average']]
df1 = df1.rename(columns={'genre_0':'genre'})

df2 = data[['genre_1','vote_average']]
df2 = df2.rename(columns={'genre_1':'genre'})

df3 = data[['genre_2','vote_average']]
df3 = df3.rename(columns={'genre_2':'genre'})
# df1.append(df2, ignore_index = True)
data_4 = pd.concat([df1, df2, df3])
data_4 = data_4.dropna()

# print(data_4)
print(data_4.shape)
print(data_4.head())

f1 = data_4['genre'].values
f2 = data_4['vote_average'].values
X = np.array(list(zip(f1, f2)))

plt.axis([0, 20, 0, 10])
plt.xlabel("Genre")
plt.ylabel("Vote Average")
plt.scatter(f1, f2, c='black', s=3)
plt.show()

kmeans = KMeans(n_clusters=15, max_iter=100)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

print(C)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', s=3)
plt.scatter(C[:, 0], C[:, 1], color='black', s=3)
plt.show()