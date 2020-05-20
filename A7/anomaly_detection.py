#!/usr/bin/env python
# coding: utf-8

# In[1]:


# anomaly_detection.py
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.mllib.linalg import VectorUDT as VUlib, Vectors as Vlib
from pyspark.ml.feature import VectorAssembler
spark = SparkSession.builder.appName('anomaly detection').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+


# In[2]:


class AnomalyDetection():

    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]),(1, ["http", "udf", 0.5]),(2, ["http", "tcp", 0.5]),(3, ["ftp", "icmp", 0.1]), (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = spark.createDataFrame(data, schema)
        
    def readData(self, filename):
        self.rawDF = spark.read.parquet(filename).cache()

    def cat2Num(self, df, indices):
        def onehotencode(x, onehot):
            return Vectors.dense(onehot[x])
        
        def delindex(x,index):
            for index in indices:
                del x[index]
            x = [float(i) for i in x] 
            return Vectors.dense(x)
        join = list()
        for index in indices:
            df = df.withColumn("catvar"+str(index),df['rawFeatures'][index])
            values = df.select("catvar"+str(index)).distinct().collect()
            onehot = dict()
            for val in values:
                onehot[val[0]] = [1.0 if x == val else 0.0 for x in values]
            onehotudf = udf(lambda x: onehotencode(x,onehot), VectorUDT())
            df = df.withColumn("onehot"+str(index), onehotudf(col("catvar"+str(index))))
            join.append("onehot"+str(index))
        indices.sort(reverse=True)
        print (indices)
        
        deleteudf = udf(lambda x: delindex(x,index),VectorUDT())
        df = df.withColumn("without",deleteudf(df["rawFeatures"]))
        join.append("without")

        assembler = VectorAssembler( inputCols=join, outputCol="sparsefeatures")
        df = assembler.transform(df)
        
        convert1udf = udf(lambda x: Vlib.dense(x.toArray()), VUlib())
        df = df.withColumn("features",convert1udf(col("sparsefeatures"))).select("id","rawFeatures","features")
        return df

    def addScore(self, df):
        pred_size = df.groupBy("prediction").count()
        nmax = pred_size.agg({"count":"max"}).collect()[0][0]
        nmin = pred_size.agg({"count":"min"}).collect()[0][0]
        scoreudf = udf(lambda x: (nmax - x)/(nmax - nmin), FloatType())
        score_df = pred_size.withColumn('score',scoreudf(col("count")))
        df = df.join(score_df,"prediction").select("id","rawFeatures","features","prediction","score")
        return df

    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        #Adding the prediction column to df1
        modelBC = spark.sparkContext.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly
        df3 = self.addScore(df2).cache()
        df3.show()

        return df3.where(df3.score > t)


# In[3]:


if __name__ == "__main__":
    ad = AnomalyDetection()
    ad.readData('data/logs-features-sample')
    anomalies = ad.detect(8, 0.97)
    print(anomalies.count())
    anomalies.show()


# In[ ]:




