# Recommender System using APACHE Spark 2.0

Implementation of Recommender Systems using APACHE Spark 2.0

This project contains following algorithm implementation:
  - Baseline Predictor
  - [Alternative Least Square Matrix Completion](http://www.grappa.univ-lille3.fr/~mary/cours/stats/centrale/reco/paper/MatrixFactorizationALS.pdf) (native pyspark.ml implementation)
  - Item-based Collaborative Filtering (To be Added)
  - User-based Collaborative Filtering (To be Added) 
  - [Funk Singular Value Decomposition with Dimensionality Reduction](https://arxiv.org/pdf/0810.3286v1.pdf) (To be Added)
  - [Matrix Completion Using Singular Value Thresholding](http://sifter.org/~simon/journal/20061211.html) (To be Added)

## Requirements

- Python 2.6 or above
- Spark 2.0 with 
  - pyspark.sql
  - pyspark.ml.recommendation
  - pyspark.ml.evaluation

## Dataset
[GroupLens Research](https://grouplens.org) has collected and made available rating data sets from the [MovieLens web site](http://movielens.org). The data sets were collected over various periods of time, depending on the size of the set. 

This project contains MovieLens 1M Dataset which contains 1 million ratings from 6000 users on 4000 movies. Please refer [README](http://files.grouplens.org/datasets/movielens/ml-1m-README.txt) page before using the data. 

## Usage

Locate the project folder on your SPARK_HOME directory. Properly modify dataPath in Recommender.py if needed. Any dataset with user-item ratings can be used here.

Main script is Recommender.py. Use spark-submit to submit this file to Spark Master.

    $ $SPARK_HOME/bin/spark-submit --master=$MASTER_URL $SPARK_HOME/recommender/Recommender.py > out

## Results

```
  recommender initialized...
  read data from 'recommender/ml-1m/ratings.dat'...
  train data...
  evalute model using test data...
  baseline RMSE = 1.11407729711
  ALS: Dropped 26 values due to null predictions (0.013020% of total data)
  ALS RMSE = 0.890409180674
```

Junyong Lee / [@espuer](https://github.com/espuer)
