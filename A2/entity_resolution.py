# entity_resolution.py
import re
import operator
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import udf, lower
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer

@udf(returnType=types.ArrayType(types.StringType()))
def split_tokens(st):
    st = st.lower()
    tokens = re.split('\W+',st)
    new_tokens = []
    for i in tokens:
        if i != '':
            new_tokens.append(i)
    return new_tokens

@udf(returnType=types.FloatType())
def jaccardsimilarity(key1,key2):
    key1 = set(key1)
    key2 = set(key2)
    similarity = float(len(key1.intersection(key2))/ len(key1.union(key2)))
    return similarity

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = spark.read.parquet(dataFile1).cache()
        self.df2 = spark.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):
        df.createOrReplaceTempView("df")
        concat_df = spark.sql("SELECT *,CONCAT_WS(' ',"+cols[0]+",' ',"+cols[1]+") AS new_col FROM df")
        token = concat_df.withColumn("tokenized",split_tokens("new_col"))
        stopword_removal = StopWordsRemover(inputCol="tokenized",outputCol="joinKey",stopWords=list(self.stopWordsBC))
        final_df = stopword_removal.transform(token).drop("new_col","tokenized")
        return final_df


    def filtering(self, df1, df2):
        df1.createOrReplaceTempView("df1")
        df2.createOrReplaceTempView("df2")
        df1_key_explode = spark.sql("SELECT id AS id1,joinKey, EXPLODE(joinKey) AS key1_explode FROM df1")
        df1_key_explode.createOrReplaceTempView("df1_key_explode")
        df2_key_explode = spark.sql("SELECT id AS id2,joinKey, EXPLODE(joinKey) AS key2_explode FROM df2")
        df2_key_explode.createOrReplaceTempView("df2_key_explode")
        candDF = spark.sql("SELECT DISTINCT d1.id1,d1.joinKey AS join_Key1,d2.id2,d2.joinKey AS join_Key2 FROM df1_key_explode d1, df2_key_explode d2 WHERE d1.key1_explode=d2.key2_explode")
        return candDF

    def verification(self, candDF, threshold):
        resultDF = candDF.withColumn("jaccard",jaccardsimilarity(candDF["join_Key1"],candDF["join_Key2"]))
        resultDF.createOrReplaceTempView("resultDF")
        resultDF = spark.sql("SELECT * FROM resultDF WHERE jaccard >= "+str(threshold))
        return resultDF

    def evaluate(self, result, groundTruth):
        T = len(set(result).intersection(groundTruth))
        precision = T / len(result)
        recall = T / len(groundTruth)
        if (precision + recall) == 0:
            FMeasure = "Invalid FMeasure"
        else:
            FMeasure = (2 * precision * recall) / (precision + recall)
        return precision,recall,FMeasure

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)
        print ("After Filtering: %d pairs left" %(candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print ("After Verification: %d similar pairs" %(resultDF.count()))

        return resultDF


    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('entity resolution').getOrCreate()
    assert spark.version >= '2.3' # make sure we have Spark 2.3+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = spark.read.parquet("Amazon_Google_perfectMapping_sample").rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
