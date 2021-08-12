
# configure pyspark
import os
import sys
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import functions as F

from pyspark.sql.types import *
from pyspark.sql.types import IntegerType, StringType

from pyspark.sql.functions import *
from pyspark import sql
from collections import defaultdict

def addRatingLabel(samples):
    total_count = samples.count()
    percentage = samples.groupBy("movieId").count().withColumnRenamed("count","movie_cnt").withColumn("Percentage",F.col("movie_cnt")/total_count)
    samples = samples.join(percentage, on=['movieId'], how='left')
    samples = samples.withColumn("label", F.when(F.col("rating")>3., 1).otherwise(0))
    return samples



def extractReleaseYearUdf(title):
    # add realease year
    if not title or len(title.strip()) < 6:
        return 1990
    else:
        yearStr = title.strip()[-5:-1]
    return int(yearStr)


def addMovieFea(movie_fea, rating_fea,round_num=2, use_MultiOneHot = False):
    #first use regular expression to convert list of genres to a list
    # then use explode function to expand the list
    
    # convert movie feature to onehot if enabled
    genres_cnt = movie_fea.withColumn("splitted_genres",F.explode(F.split(F.col('genres'), "\\|"))).groupBy('movieId').count()
    genres_cnt = genres_cnt.withColumnRenamed("count", "genres_cnt")
    
    movie_fea = movie_fea.join(genres_cnt, on="movieId", how="left")
    
    if use_MultiOneHot: 
        tmp = movie_fea.withColumn("splitted_genres",F.explode(F.split(F.col('genres'), "\\|"))).drop("genres")
        multi_onehot = tmp.groupBy("movieId").pivot("splitted_genres").count().fillna(0)
        # rename columns
        for c in multi_onehot.columns:
            if 'movie' not in c:
                multi_onehot = multi_onehot.withColumnRenamed(c, "genres_"+c)
        #multi_onehot.show()
        samples = movie_fea.drop('genres').join(multi_onehot, on= "movieId", how= "left")
    else:
        samples = movie_fea.withColumn("movieGenre1",F.split(F.col('genres'),"\\|")[0])\
                            .withColumn("movieGenre2",F.split(F.col('genres'),"\\|")[1])\
                            .withColumn("movieGenre3",F.split(F.col('genres'),"\\|")[2])
        #samples = movie_fea
        
    
    samples = rating_fea.join(samples, on=['movieId'], how='left')
    # add releaseYear,title
    samples = samples.withColumn('releaseYear',
                                                       F.udf(extractReleaseYearUdf, IntegerType())('title')) \
        .withColumn('title', F.udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
        .drop('title')
    
    
        
    # compute statistic for each movie: count, avg rating, std rating
    movie_stat = rating_fea.groupBy("movieId").agg(F.count(F.lit(1)).alias("movieRatingCount"), 
                                              F.format_number(F.avg(F.col("rating")), round_num).alias("movieAvgRating"), 
                                              F.format_number(F.stddev(F.col("rating")), round_num).alias("movieStdRating") ).fillna(0.)
    movie_fea = samples.join(movie_stat, on=["movieId"], how="left")
    
    return movie_fea
    
    





def extractSortedGenres(genres_list):
    """
    input: a list of concatenated genres string like ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    output: a list of genres sorted by frequency of genre ['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    example:
        if we have a list of (genre, frequency) ,like (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
        then we sort it in descending order and return ['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    """
    genre_ls = defaultdict(int) 
    for genres in  genres_list:
        for genre in genres.split('|'):
            genre_ls[genre] += 1
    # genre_ls.item() = (key=genre, value=count)        
    # return sorted list, not dictionary!
    sorted_genres = sorted(genre_ls.items(), key=lambda x:x[1], reverse=True )
    # return list of genre
    return [ g[0] for g in sorted_genres]
    
    


def addUserFea(samplesWithMovieFea, round_number = 2):
    """
    input:
        samplesWithMovieFea: Spark DataFrame with movie features
        round_num: precision number 
    output:
        dataframe with extracted user features
    """
    # extract behavior features
    extractSortedGenres_udf = F.udf(extractSortedGenres, ArrayType(StringType()))
    # add user statistic: Rating count, AverageRating, Rating Stddev,  AverageReleaseYear, ReleaseYearStddev
    # use window function to add new feature column and each user has the same value in this column
    # the first line equivalent to   select count() over (partition by userId, order by timestemp)
    
    #    samplesWithUserFea.filter(samplesWithMovieFea['userId'] == 1).orderBy(F.col('timestamp').asc()).show(truncate=False)
    #   samplesWithUserFea.where(F.col("userId") == 2).show()
    
    
    #  Behavior data:
    #  The genres each user visits/likes most frequently (we can choose top k): it tells user's daily hobbies
    #  The genres each user visits/likes recently according to timestamp: it tells how user's prefernce changes based on given genres
    #  The movies each user visits/likes recently:
    
    
    samples = samplesWithMovieFea.withColumn("userRatingCnt", F.count(F.lit(1))\
                                             .over(sql.Window.partitionBy('userId')\
                                                   .orderBy('timestamp').rowsBetween(-100,-1))) \
                                 .withColumn("userAvgRating", format_number(F.avg("rating")\
                                             .over(sql.Window.partitionBy("userId")\
                                                   .orderBy('timestamp').rowsBetween(-100,-1)), round_number))\
                                 .withColumn("userRatingStddev", format_number(F.stddev("rating")\
                                             .over(sql.Window.partitionBy("userId")\
                                                   .orderBy('timestamp').rowsBetween(-100,-1)), round_number)) \
                                 .withColumn("userReleaseYearStddev", format_number(F.stddev("releaseYear")\
                                             .over(sql.Window.partitionBy("userId")\
                                                   .orderBy('timestamp').rowsBetween(-100,-1)), round_number)) \
                                 .withColumn("userAvgReleaseYear", format_number(F.avg("releaseYear")\
                                             .over(sql.Window.partitionBy("userId")\
                                                   .orderBy('timestamp').rowsBetween(-100,-1)), round_number).cast(IntegerType()))\
                                 .withColumn("userActiveMovies", F.collect_list(when(F.col("label")==1, F.col("movieId")).otherwise(F.lit(None)))\
                                             .over(sql.Window.partitionBy("userId") \
                                             .orderBy("timestamp").rowsBetween(-100,-1)))\
                                 .withColumn("userRatedMovie1", F.col("userActiveMovies")[0])\
                                 .withColumn("userRatedMovie2", F.col("userActiveMovies")[1])\
                                 .withColumn("userRatedMovie3", F.col("userActiveMovies")[2])\
                                 .withColumn("userRatedMovie4", F.col("userActiveMovies")[3])\
                                 .withColumn("userRatedMovie5", F.col("userActiveMovies")[4])\
                                 .withColumn("userGenres", extractSortedGenres_udf(F.collect_list(when(F.col('label') == 1, F.col('genres')).otherwise(F.lit(None)))\
                                             .over(sql.Window.partitionBy("userId")\
                                                   .orderBy('timestamp').rowsBetween(-100,-1))))\
                                 .withColumn("userGenre1", F.col("userGenres")[0])\
                                 .withColumn("userGenre2", F.col("userGenres")[1])\
                                 .withColumn("userGenre3", F.col("userGenres")[2])\
                                 .withColumn("userGenre4", F.col("userGenres")[3])\
                                 .withColumn("userGenre5", F.col("userGenres")[4])\
                                 .drop("userActiveMovies","userGenres","genres")\
                                 .filter(F.col("userRatingCnt")>1) # remove the  users who watch movies once or even don't watch movie
    samples.printSchema()
    samples.show(5)

    return samples



def SampleTrainTestDataByTime(samples, sample_rate=0.1, save_path ="../../data/processed_data/"):
    """
    This function is to sample a small amount of samples from the huge dataset,
    then it splits the sampled dataset according timestamp
    For example, in a range of timestamp from 1second to 10000 second,80% samples are before 1000sec 
    and 20% samples are after 1000sec, we taks those 80% samples as training set and 20% samples as test set.
    This simulates the real world setting: we use data collected before a date and use data collected after this date
    to test the model.
    
    """
    samples = samples.sample(sample_rate).withColumn("timestamplong", F.col("timestamp").cast(LongType()))
    # approximate 80% quantile with 0.05 tolerance
    quantile = samples.stat.approxQuantile("timestamplong", [0.8], 0.05)
    timestamp_boundary = quantile[0]
    training_samples = samples.where(F.col("timestamplong")<=timestamp_boundary).drop("timestamplong")
    test_samples = samples.where(F.col("timestamplong")>timestamp_boundary).drop("timestamplong")
    train_file = save_path + "train.csv"
    test_file = save_path + "test.csv"
    
    # save files
    # repartition(1) is to amke all saved samples in the same csv file
    training_samples.repartition(1).write.option("header","true").mode("overwrite").csv(train_file)
    test_samples.repartition(1).write.option("header","true").mode("overwrite").csv(test_file)
    training_samples.toPandas().to_csv(test_file,header=True,index=False)
    test_samples.toPandas().to_csv(test_file,header=True,index=False)
    ## or equivalently
    # training_samples.write.csv(train_file,header=True, mode="overwrite")
    # test_samples.write.csv(test_file,header=True, mode="overwrite")
    return



if __name__=="__main__":
    os.environ['PYSPARK_DRIVER_PYTHON']="/home/wenkanw/.conda/envs/mlenv/bin/python3" # path to python exec file
    os.environ['PYSPARK_PYTHON']="/home/wenkanw/.conda/envs/mlenv/bin/python3" #path to python exec file
    data_path = "../../data/"

    # Spark configuration
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    # Create Spark instance
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    movie_df = spark.read.csv(data_path+"movies.csv", header=True)
    movie_df.printSchema()
    #movie_df.show(5)
    link_df = spark.read.csv(data_path+"links.csv", header=True)
    link_df.printSchema()
    #link_df.show(5)

    rating_df = spark.read.csv(data_path+"ratings.csv", header=True)
    rating_df.printSchema()
    #rating_df.show(5)

    label_df = addRatingLabel(rating_df)
    #label_df.show()
    movie_fea = addMovieFea(movie_df, label_df,round_num=2)
    movie_fea.show(10)
    transformed_samples = addUserFea(movie_fea)
    SampleTrainTestDataByTime(transformed_samples, sample_rate=0.1, save_path ="../../data/processed_data/")
    
    
    
