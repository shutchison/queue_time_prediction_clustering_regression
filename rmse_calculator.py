import sys
import os
import datetime
import pyspark
import math

from functools import reduce  # For Python 3.x

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql.types import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.sql.functions import col

spark = SparkSession.builder.master("local").appName("ml_pipeline").getOrCreate()
sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")

if len(sys.argv) < 2:
    print("Send in one argument: the number of clusters to build and assess.")
    exit(1)
# Load saved ml data from file

ml_input_schema = StructType([StructField("Submit",TimestampType(),True),
                              StructField("BeocatCPUsInUse",IntegerType(),True),
                              StructField("BeocatMemoryInUse",IntegerType(),True),
                              StructField("max(QueueDepth)",IntegerType(),True),
                              StructField("ReqCPUs",IntegerType(),True),
                              StructField("ReqMem",IntegerType(),True),
                              StructField("ReqMinutes",IntegerType(),True),
                              StructField("OwnsResources",BooleanType(),True),
                              StructField("QueueTimeInSec",IntegerType(),True)])

ml_input_df = spark.read.csv(path=r"/homes/scotthutch/825_ml_input_df_with_req_minutes/ml_input_save.csv", schema=ml_input_schema)
# UDF for converting column type from vector to double type
unlist = udf(lambda x: float(list(x)[0]), DoubleType())

assembler = VectorAssembler(inputCols=["BeocatCPUsInUse"], outputCol="BeocatCPUsInUse_Vect")
thing = assembler.transform(ml_input_df)

scaler = MinMaxScaler(inputCol="BeocatCPUsInUse_Vect", outputCol="BeocatCPUsInUse_Scaled", max=100.0)
thing2 = scaler.fit(thing).transform(thing).withColumn("BeocatCPUsInUse_Scaled", unlist("BeocatCPUsInUse_Scaled")).drop("BeocatCPUsInUse_Vect")

assembler = VectorAssembler(inputCols=["BeocatMemoryInUse"], outputCol="BeocatMemoryInUse_Vect")
thing3 = assembler.transform(thing2)

scaler = MinMaxScaler(inputCol="BeocatMemoryInUse_Vect", outputCol="BeocatMemoryInUse_Scaled", max=100.0)
thing4 = scaler.fit(thing3).transform(thing3).withColumn("BeocatMemoryInUse_Scaled", unlist("BeocatMemoryInUse_Scaled")).drop("BeocatMemoryInUse_Vect")

assembler = VectorAssembler(inputCols=["BeocatCPUsInUse_Scaled",
                                       "BeocatMemoryInUse_Scaled",
                                       "max(QueueDepth)",
                                       "ReqCPUs",
                                       "ReqMem",
                                       "ReqMinutes",
                                       "OwnsResources"], outputCol="features")

thing5 = assembler.transform(thing4).select("features", "QueueTimeInSec")

#get rid of strange first entry
data = thing5.filter(~(thing5.QueueTimeInSec == 70443))
data = data.withColumn("label", data.QueueTimeInSec.cast("Double")).drop("QueueTimeInSec")
#using cross validation to build the base model

# seed set for reproducability
train_data, test_data = data.randomSplit([.8, .2], seed=2)
print("Number in train_data = {}".format(train_data.count()))
print("Number in test_data = {}".format(test_data.count()))

gbt = GBTRegressor(featuresCol="features", predictionCol="gbt_prediction", maxIter=150)
grid = ParamGridBuilder().baseOn({gbt.featuresCol: "features"}).build()
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="gbt_prediction",
                                metricName="rmse")

cv = CrossValidator(estimator=gbt,
                    estimatorParamMaps=grid,
                    evaluator=evaluator,
                    numFolds=5,
                    parallelism=1,
                    seed=2)

num_clusters = int(sys.argv[1])
print("Trying to cluster into {} clusters".format(num_clusters))

base_dir = r"/homes/scotthutch/825_distributed_code/models"
num_in_test_data = test_data.count()

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

base_model = CrossValidatorModel.load(os.path.join(base_dir, "gbt_base"))
print("base model loaded")
kmeans_model = KMeansModel.load(os.path.join(base_dir, "k_means_{}_clusters".format(num_clusters)))

print("k means model for {} cluters loaded".format(num_clusters))
gbt_models = []
for i in range(num_clusters):
    current_model_dir = os.path.join(base_dir, "k_means_{}_clusters_gbt_model_{}".format(num_clusters, i))
    if(os.path.isdir(current_model_dir)):
        model = CrossValidatorModel.load(current_model_dir)
        gbt_models.append(model)
        #print("Model {} loaded".format(i))
    else:
        gbt_models.append(base_model)
        #print("Model {} not found.  Using base model".format(i))
print("{} models loaded".format(len(gbt_models)))

cluster_pred = kmeans_model.transform(test_data)
clusters = []
for i in range(num_clusters):
    cluster = cluster_pred.filter(col("prediction")==i)
    gbt_pred = gbt_models[i].transform(cluster)
    clusters.append(gbt_pred)

print("Predictions made.  Evaluating.")
all_gbt_preds = unionAll(*clusters)
squared_error = all_gbt_preds.withColumn("squared_error", (col("label") - col("gbt_prediction"))**2)
squared_error_sum = squared_error.agg(F.sum(col("squared_error"))).collect()[0][0]
rmse = math.sqrt(squared_error_sum/num_in_test_data)
print("For {} clusters, rmse is {}\n".format(num_clusters, rmse))
