# NTU-20191018
## NTU CZ4032 Data Analytics & Mining Assignment
### Authors: Li Bingzi, Wu Ziang, Zhang Shengjing, and Zhang Yuehan

### Dataset: Kaggle TMDb 5000 Movie Dataset
https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv  

Third party libraries (install via python pip):

numpy == 1.17.2

pandas == 0.25.1

matplotlab == 3.1.1

seaborn == 0.9.0

sklearn == 0.0

nltk == 3.4.5

apyori == 1.1.1

The source code python script or jupyter notebook scripts can be found in the repository. These scripts are independent and the explanation of the scripts are given below:

1. data_preprocessing.ipynb performs data integration

2. data_selection.ipynb performs data cleaning, feature creation, feature subset selection and some data analysis

3. feature_elimination.ipynb performs some data analysis, completes the feature selection and most importantly outputs a data with 4790 samples each of 29 features

4. data_analytics.ipynb performs data analysis, including summary statistics, correlation and visualization, and implement the moive recommendation application

5. data_bucketization.ipynb prepares data for data mining by bucketization and vectorization

6. data_train_test.ipynb prepares training and test data for classifiers

7. classification.ipynb trains K-Nearest Neighbors, Gaussian Naïve Bayes, Support Vector Machines, Artificial Neural Networks, and Ensemble Classifier to predict the popularity and evaluates their performance with Accuracy Score, Confusion Matrix, Precision, and ROC Curve.

8. clustering_1.py applies K-Means clustering to explore the relations between features, popularity, vote counts, vote rate and runtime.

9. clustering_2.ipynb applies DESCAN and Agglomerative to explore some implicit grouping of movies

10. association_rule.ipynb applies Apriori to explore the assocation between genres and keywords

The sample output are in the directory output and are explained below:

1. data/tmdb_5000_credits.csv and data/tmdb_5000_features.csv are the original dataset from Kaggle

2. data/tmdb_5000_movies.csv is the dataset after data preprocessing

3. figure directory contains all figures generated from the scripts and used in the report

4. notebook directory contains the html version of the jupyter notebook exported from the local server

5. output directory contains all intermediate outputs, for example dataset after data intergration, training and test data after bucketization and vectorization


