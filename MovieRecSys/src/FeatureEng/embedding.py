
def getItemSeqs(spark, samplesRating):
    """
    extract item sequences for each user from dataframe
    1. for each user, collect the corresponding visited movies and timestamp into a list
    2. use UDF to process movie list and timestamp list to sort the movie sequence for each user
    3. join the movie list to get a string for each user
    """
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        # concat timestamp with movie id
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]
    
    
    sortUDF = udf(sortF, ArrayType(StringType()))
    
    # rating data
    #ratingSamples.show(5)
    # ratingSamples.printSchema()
    userSequence = samplesRating.where(F.col("rating") > 3) \
                    .groupBy("userId")\
                    .agg(sortUDF(F.collect_list("movieId"), F.collect_list("timestamp")).alias("movieIds"))\
                    .withColumn("movieIdStr", F.array_join(F.col("movieIds"), " "))
    seq = userSequence.select("movieIdStr").rdd.map(lambda x : x[0].split(" "))
    #print(seq.collect()[:5])
    return seq



def embeddingLSH(spark, movieEmbMap):
    """
    Local sensitive hashing using bucketedRandomProjection
    """
    movieEmbSeq = []
    for key, embedding_list in movieEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in embedding_list]
        movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb", outputCol="bucketId", bucketLength=0.1,
                                                      numHashTables=3)
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    embBucketResult = bucketModel.transform(movieEmbDF)
    print("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    print("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate=False)
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate=False)



def getTransitionMatrix(item_seq):
    """
    build graph and transition matrix based on input item sequences 
    input: list of item sequence in RDD format
    output: transition matrix and item distribution in dictionary format
    
    """
    def generate_pair(x):
        """
        use a sliding window with size of 2 to generate item pairs
        input: ls =  list of items 
        output: list of pairs
        example:
            input: [86, 90, 11, 100,]
            output: [[86,90], [90, 11], [11,100]]
        """
        res = []
        prev = None
        print(x)
        for i in range(len(x)):
            if i >0:
                res.append((x[i-1],x[i]))
        return res

    
    #  convert item sequences to pair list 
    pair_seq = item_seq.flatMap(lambda x: generate_pair(x))
    # convert pair list to  dictionary, key = pair, value = count
    pair_count_dict = pair_seq.countByValue()
    tot_count = pair_seq.count()
    trans_matrix =  defaultdict(dict)
    item_count = defaultdict(int)
    item_dist = defaultdict(float)
    
    # consider out-degree only 
    for item, cnt in pair_count_dict.items():
        item1, item2 = item[0], item[1]
        item_count[item1] +=  cnt
        trans_matrix[item1][item2] = cnt
        
    for item, cnt in pair_count_dict.items():
        item1, item2 = item[0], item[1]
        # possibility of transition
        trans_matrix[item1][item2] /= item_count[item1]
        # distribution of each source node (item)
        item_dist[item1] =  item_count[item1]/tot_count
        
    return trans_matrix, item_dist

def oneRandomWalk(trans_mat, item_dist, sample_length):
    """
    generate one random walk sequence based on transition matrix
    input: 
        - trans_mat: transition matrix
        - item_dist: distribution of item
        - sample length: number of node in a path  = length of a walk -1 = length of edges - 1
    """
    rand_val = random.random()
    # randomly pick item based on CDF , cumulative density function, obtained from the item distribution
    # we can also randomly pick a item based on the distribution using  choice () function from numpy as well
    cdf_prob =0
    first_item = ''
    for item, prob in item_dist.items():
        cdf_prob += prob
        if cdf_prob >= rand_val:
            first_item = item
            break
    item_list = [first_item]
    cur_item = first_item
    
    while len(item_list) < sample_length:
        if (cur_item not in item_dist) or (cur_item not in trans_mat):
            break
        cdf_prob = 0
        rand_val = random.random()
        dist = trans_mat[cur_item]
        for item, prob in dist.items():
            cdf_prob += prob
            if cdf_prob >= rand_val:
                cur_item = item
                break
        item_list.append(cur_item)
        
    return item_list



def generateItemSeqs(trans_mat, item_dist, num_seq=20000, sample_length = 10  ):
    """
    use random walk to generate multiple item sequences
    """
    samples = []
    for i in range(num_seq):
        samples.append(oneRandomWalk(trans_mat, item_dist, sample_length))
    
    return samples


def trainItem2Vec(spark, item_seqs, emb_length, output_path, save_to_redis=False, redis_keyprefix=None):
    """
    use Word2Vec to train item embedding
    input:
        - item_seqs: RDD pipeline instance, rather than dataframe
    Note:  
    - Word2Vec from mllib is a function that take RDD pipeline as input.
    - Word2Vec from ml is a function that take Dataframe as input 
    
    """
    # train word2Vec
#     w2v = Word2Vec(vectorSize=emb_length, windowSize = 5, maxIter = 10, seed=42)
    w2v = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = w2v.fit(item_seqs)
    # test word2vec
    synonyms = model.findSynonyms("157", 20)
    for synonym, cos_similarity in synonyms:
        print(synonym, cos_similarity)
    
    # save word2Vec to input path 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path, "w") as fp:
        for movie_id in model.getVectors():
            # convert vector to string type and store it
            vector = " ".join([str(emb) for emb in model.getVectors()[movie_id]])
            pair = movie_id + ":" + vector + "\n"
            fp.write(pair)
    return model


def getDeepWalk(spark, item_seq, sample_length=10, num_walk=20000, output_file='../../data/modeldata/embedding.csv',
             save_to_redis=False, redis_key_prefix=None):
    """
    use DeepWalk to generate graph embeddings
    input:
        - item_seq: RDD based sequence of item visited by a user
        
    """
    
    # construct probability graph
    trans_mat, item_dist = getTransitionMatrix(item_seq)
    
    # generate sequence samples randomly
    samples = generateItemSeqs(trans_mat, item_dist,num_seq=num_walk, sample_length = sample_length )
    # convert list of samples to spark rdd 
    samples_rdd = spark.sparkContext.parallelize(samples)
    # train item2Vec
    graphEmbModel = trainItem2Vec(spark, samples_rdd, emb_length=10, output_path=output_file , save_to_redis=False, redis_keyprefix=None)
    
    return graphEmbModel

def getUserEmb( spark ,samples_rating, item_emb_model, output_file):
    """
    generate user embedding based on item embedding
    use map reduce to sum up embeddings of items purchased by user to generate user embedding
    input:
        - spark: spark session
        - samples_rating: dataframe with rating, movieId, userId data
        - item_emb_model: word2Vec/Item2Vec model trained by deep walk. 
        - output_file: file name of user embedding 
    
    """
    
#     assert not item_emb or not item_emb_path, "Must input either item embedding vectors or path"
#     if item_emb_path != None:
#         item_emb = spark.read.csv(item_emb_path, header=True)

    emb_dict = item_emb_model.getVectors()
    item_emb_ls=[]
    for item, emb in emb_dict.items():
        #print((item, emb))
        item_emb_ls.append((item, list(emb)))
    fields = [StructField('movieId', StringType(),False),
             StructField('emb', ArrayType(FloatType()),False),]
    item_emb_schema = StructType(fields)
    item_emb_df = spark.createDataFrame(item_emb_ls, item_emb_schema)
    
    # apply mapreduce to sum up item embeddings for each user to obtain user embedding
    # Note: we need inner join here to avoid empty item embedding during mapreduce calculation
    user_emb = samples_rating.join(item_emb_df, on="movieId", how="inner")
    print()
    print("User Embdding")
    user_emb.show(5)
    user_emb.printSchema()
    user_emb = user_emb.select("userId","emb").rdd.map(lambda row: (row[0], row[1]) ).reduceByKey(lambda emb1, emb2: [ float(emb1[i]) + float(emb2[i]) for i in range(len(emb1))] ).collect()
    print(user_emb[:5])
    #save user embedding
    with open(output_file,"w") as fp:
        for userId, emb in user_emb:
            row = " ".join([str(e) for e in emb])
            row = str(userId)+ ":"+ row + "\n"
            fp.write(row)
    print("User Embedding Saved!")
    return


if __name__ == '__main__':
    conf = SparkConf().setAppName('ctrModel').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # Change to your own filepath
    file_path = '../../data/'
    rawSampleDataPath = file_path + "ratings.csv"
    embLength = 10
    print("Process ItemSquence...")
    samplesRating = spark.read.csv(rawSampleDataPath, header = True)
    item_seqs = getItemSeqs(spark, samplesRating)
    #print(samples)
    
    #trainItem2Vec(item_seqs, emb_length=10, output_path=file_path+"modeldata/itemGraphEmb.csv", save_to_redis=False, redis_keyprefix=None)
    
    graphEmb = getDeepWalk(spark, item_seqs, sample_length=10, num_walk=20000, output_file=file_path+"modeldata/itemGraphEmb.csv",
             save_to_redis=False, redis_key_prefix=None)
    getUserEmb( spark ,samples_rating= samplesRating, item_emb_model= graphEmb, output_file= file_path+"modeldata/userEmb.csv")

    print("Done!")
   
