from __future__ import print_function
import sys
if sys.version >= '3':
   long = int

from pyspark.sql import SparkSession, Row
from pyspark.ml.evaluation import RegressionEvaluator
from algorithm import BaseLine, ALS

class Recommender:
   def __init__(self):
      self.spark = None
      self.trainDF = None
      self.testDF = None
      self.models = {}

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
      ratings = self.spark.createDataFrame(ratingsRDD).cache()
      (self.trainDF, self.testDF) = ratings.randomSplit([0.8, 0.2])

   def train(self, algorithm="baseline"):
      if algorithm == "baseline":
         self.models["baseline"] = BaseLine().train(self.trainDF)
         return

      if algorithm == "ALS":
         self.models["ALS"] = ALS().train(self.trainDF)
         return

      print("unknown alogrithm for train: " + algorithm)

   def evaluateModel(self, algorithm="baseline"):
      predictions = None
      if algorithm == "ALS":
         predictions = ALS().predict(self.models["ALS"], self.testDF)

      if algorithm == "baseline":
         predictions = BaseLine().predict(self.models["baseline"], self.testDF)

      if predictions is None:
         print("unknown alogrithm for evaluateModel: " + algorithm)
         return
      # evaluate result using RMSE metric
      evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                      predictionCol="prediction")
      rmse = evaluator.evaluate(predictions)
      print(algorithm + " RMSE = " + str(rmse))

#dataPath = "data/mllib/als/sample_movielens_ratings.txt"
dataPath = "rc/ml-1m/ratings.dat"
algorithms = ["baseline", "ALS"]

if __name__ == "__main__":
   print("recommender initialized...")
   with Recommender() as recommender:
      print("read data from '" + dataPath + "'...")
      recommender.readData(dataPath)

      print("train data... ")
      for algo in algorithms:
         recommender.train(algo)

      print("evalute model using test data... ")
      for algo in algorithms:
         recommender.evaluateModel(algo)
      
