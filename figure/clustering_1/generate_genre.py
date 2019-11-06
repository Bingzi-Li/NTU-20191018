import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import csv
import itertools

plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('seaborn-whitegrid')

# Importing the dataset
data = pd.read_csv('/path/to/data/tmdb_5000_features.csv', usecols = ['genre_0','genre_1','genre_2'])
df = pd.read_csv('/path/to/data/tmdb_5000_features.csv')

data = np.array(data)
data_list = data.tolist()
data_list = list(set(list(itertools.chain.from_iterable(data_list))))
genre = [x for x in data_list if str(x) != 'nan']
print(genre)

for i in genre:
    df['genre_0'].replace(i, genre.index(i), inplace=True)
    df['genre_1'].replace(i, genre.index(i), inplace=True)
    df['genre_2'].replace(i, genre.index(i), inplace=True)
# print(df)
df.to_csv(r'/path/to/data/genre.csv')
