from __future__ import print_function
import sys
if sys.version >= '3':
   long = int

from math import sqrt
from pyspark.sql import SparkSession, Row
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

class Recommender:
   def __init__(self):
      self.spark = None
      self.trainData = None
      self.testData = None
      self.model = None
      self.ratings = None
      self.models = {}
      self.predictions = {}

   def __enter__(self):
      self.spark = SparkSession\
                  .builder\
                  .appName("Recommender")\
                  .getOrCreate()
      return self

   def __exit__(self, exc_type, exc_value, traceback):
      if exc_type is not None:
         print(str(exc_type) + str(exc_value) + str(traceback))

      if self.spark is not None:
         self.spark.stop()

   def readData(self, path):
      lines = self.spark.read.text(path).rdd
      parts = lines.map(lambda row: row.value.split("::"))
      ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                           rating=float(p[2]),
                                           timestamp=long(p[3])))
      self.ratings = self.spark.createDataFrame(ratingsRDD).cache()
      (self.trainData, self.testData) = self.ratings.randomSplit([0.8, 0.2])

   def train(self, algorithm='baseline'):
      if algorithm == 'ALS':
         model[algorithm] = ALS.train(self.trainData)


   def evaluateModel(self, algorithm='baseline'):
      
      # Because random split [0.8, 0.2] could make some users in test data
      # has no rating data for their recommendations - which makes ALS predict
      # 'null' in prediction, we drop those data. Those data should be less
      # than 0.1% and hence not affect test rmse.
      totalCount = self.testData.count()
      predictions = self.model.transform(self.testData).dropna()
      dropCount = totalCount - predictions.count()
      print("ALS: Dropped {} values due to null predictions ({:f}% of total data)"\
           .format(dropCount, float(dropCount) / totalCount * 100))

      evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                      predictionCol="prediction")
      rmse = evaluator.evaluate(predictions)
      print("RMSE = " + str(rmse))

def computeRmse(testDF):
   n = testDF.count()
   return sqrt(testDF.map(lambda x: (x[1] - x[4]) ** 2).reduce(add) / float(n))

dataPath = "rc/ml-1m/ratings.dat"
#dataPath = "rc/somedata.dat"
#dataPath = "data/mllib/als/sample_movielens_ratings.txt"

if __name__ == "__main__":
   print("recommender initialized...")
   with Recommender() as recommender:
      print("read data from '" + dataPath + "'...")
      recommender.readData(dataPath)

      print("train data... ")
      recommender.train()

      print("evalute model using test data... ")
      recommender.evaluateModel()
      
      #print("RMSE = " + computeRmse(recommender.predictedTestDF))
      #print("RMSE = " + str(recommender.rmse))
