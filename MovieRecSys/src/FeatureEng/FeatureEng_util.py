
from pyspark import SparkConf, SparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import *


def OneHotEncoding(samplesWithMovieFea):
    """
    convert movieId to OneHot representation
    
    Note: 
        1. input column to onehot must be numeric type
        2. onehot output column format:  (number of onehot columns, [one_hot position], [value]), or (number of onehot columns, {position:value})
    """
    # convert movieId to int type
    samplesWithMovieFea = samplesWithMovieFea.withColumn("movieId", F.col("movieId").cast(IntegerType()))
    enc = OneHotEncoder(inputCol="movieId", outputCol="movieIdVector",dropLast=False)
    samplesWithMovieFea = enc.fit(samplesWithMovieFea).transform(samplesWithMovieFea)
    samplesWithMovieFea.show(5)
    
    return samplesWithMovieFea

def Array2Vec(genreIndices, indexSize):
    """
    Convert Array type feature to a vector
    input:
        input are the features in one row. These features include:
            genreIndices: a list of numeric indice of genres
            indexSize: size of index
    output: 
        converted multi-onehot array for each row in table
    """
    genreIndices.sort()
    # a list of values to fill into muulti-one-hot array
    fill_list = [1.0 for i in range(len(genreIndices)) ]
    
    # sparse take  size of vector, positions in vector to fill, values used to fill these position
    return Vectors.sparse(indexSize, genreIndices, fill_list)


def MultiOneHotEncoding(samplesWithMovieFea):
    """
    convert genres of each movie to multi-onehot represenatation
    Step:
        1. first explode the genres for each movie into multiple rows
        2. use StringIndexer to convert genres into genreIndex in numerical label format
        3. obtain size of index
        4. aggregate the genreIndex list to get genreIndex array list for each movie 
        5. Convert genreIndex array list into multi-onehot
    """
    # split genre list for each movie and convert string to string list. Then explode each genre into one row
    samplesWithMovieFea = samplesWithMovieFea.withColumn("genre", F.explode(F.split("genres","\\|").cast(ArrayType(StringType()))))
    # Convert string list to numerical index label list in each row
    indexer = StringIndexer(inputCol="genre",outputCol= "genreIndex")
    indexModel= indexer.fit(samplesWithMovieFea)
    samplesWithMovieFea = indexModel.transform(samplesWithMovieFea)
    print("String labels: ")
    print(indexModel.labels)
    #indexer.save("StringIndices.csv")
    # obtain size of index and add indexSize to a new column
    indexSize = samplesWithMovieFea.agg(F.max("genreIndex")).head()[0] + 1
    
    samplesWithMovieFea = samplesWithMovieFea.groupBy("movieId").agg(F.collect_list("genreIndex").alias("genreIndices"))
    samplesWithMovieFea = samplesWithMovieFea.withColumn("IndexSize", F.lit(indexSize))
    
    # Convert index list in each row into multi-onehot vector
    Array2Vec_UDF = udf(Array2Vec, VectorUDT())
    samplesWithMovieFea = samplesWithMovieFea.withColumn("MultiOneHot_Vector", Array2Vec_UDF(F.col("genreIndices"), F.col("IndexSize")) ).orderBy("movieId")
    
    
    return samplesWithMovieFea


def MultiOneHot_v2(sampleWithMovieFea):
    """
    This function convert genre array list to Multi-OneHot using Pivot function
    Use Pivot and array method to convert genres to multi-onehot, without using StringIndexer
    """
    def arr2vec(arr):
        indexSize = len(arr)
        pos = [i for i in range(len(arr)) if arr[i]==1 ]
        fill_ls = [1]*len(pos)
        return Vectors.sparse(indexSize, pos, fill_ls)
    
    arr2vec_udf = udf(arr2vec, VectorUDT())
    tmp = sampleWithMovieFea.withColumn("splitted_genres",F.explode(F.split(F.col('genres'), "\\|"))).drop("genres")
    genre_ls = [s.splitted_genres for s in  tmp.select("splitted_genres").distinct().collect()]
    genre_ls.sort()
    print(genre_ls )
    
    multi_onehot = tmp.groupBy("movieId").pivot("splitted_genres").count().fillna(0)
    ##rename columns
    #for c in multi_onehot.columns:
    #    if 'movie' not in c:
    #        multi_onehot = multi_onehot.withColumnRenamed(c, "genres_"+c)
    #multi_onehot.show()
    
    columns = [F.col(c) for c in genre_ls]
    multi_onehot = multi_onehot.withColumn("Genres_Vector",arr2vec_udf(F.array(genre_ls))).drop(*genre_ls)
    samples = sampleWithMovieFea.drop('genres').join(multi_onehot, on= "movieId", how= "left").orderBy("movieId")
    return samples

def ratingDiscretizer(samplesWithRating):
    """
    This function adds statistic features of rating and
    discretize the rating feature using Binning. Then it normalize rating feature
    using MinMaxScalar
    """
    samplesWithRating.printSchema()
    # compute statistic for each movie
    samplesWithRating  = samplesWithRating.groupBy("movieId").agg(F.avg("rating").alias("AvgRating"),
                                                                  F.variance("rating").alias("RatingVar"),
                                                                 F.count(F.lit(1)).alias("ratingCnt"))
    
    # we need to convert average rating value to a dense Vector with User Define Type (UDT)
    # udf(lambda x: Vectors.dense(x), VectorUDT()):  take  dense vector as input,  VectorUDT() type vector as output
    samplesWithRating = samplesWithRating.withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))
    print()
    print("Dense Vector")
    samplesWithRating.show(5)
    
    #Bucket and discretize the rating
    ratingDiscretizer = QuantileDiscretizer(inputCol="ratingCnt", outputCol="ratingCntBucket", numBuckets= 100)
    
    # MinMaxScaler
    scaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="ScaledAvgRating")
    pipe = Pipeline(stages = [ratingDiscretizer, scaler])
    TransformedSamples = pipe.fit(samplesWithRating).transform(samplesWithRating)
    TransformedSamples.show(5)
    
    return TransformedSamples

def test(data_path):
    # Spark configuration
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    # Create Spark instance
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    movie_df = spark.read.csv(data_path+"movies.csv", header=True)
    movie_df = OneHotEncoding(movie_df)
#     movie_df = MultiOneHot_v2(movie_df)
    movie_df = MultiOneHotEncoding(movie_df)
    movie_df.printSchema()
    movie_df.show(5)
    
    
    #movie_df.show(5)
    link_df = spark.read.csv(data_path+"links.csv", header=True)
    link_df.printSchema()
    #link_df.show(5)

    rating_df = spark.read.csv(data_path+"ratings.csv", header=True)
    rating_df.printSchema()
    #rating_df.show(5)
    rating_df = ratingDiscretizer(rating_df)
    rating_df.printSchema()


if __name__ == "__main__":
#     os.environ['PYSPARK_DRIVER_PYTHON']="/home/wenkanw/.conda/envs/mlenv/bin/python3" # path to python exec file
#     os.environ['PYSPARK_PYTHON']="/home/wenkanw/.conda/envs/mlenv/bin/python3" #path to python exec file
    data_path = "../../data/"
    test(data_path)
    
    
    
