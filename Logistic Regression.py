'''
We are creating a click-through rate (CTR) prediction pipeline. 
Dataset is Criteo Labs dataset that was used for a 2014 Kaggle competition.
https://www.kaggle.com/c/criteo-display-ad-challenge

Steps are as follows:
Step 1: Featurize categorical data using one-hot-encoding (OHE)
Step 2: Construct an OHE dictionary
Step 3: Parse CTR data and generate OHE features & Visualize Feature frequency
Step 4: CTR prediction and logloss evaluation & Visualize ROC curve
Step 5: Reduce feature dimension via feature hashing


We First convert categorical features to numerical ones using One Hot encoding (OHE)

Data points can typically be represented with a small number of non-zero OHE features relative to the total number of features that occur in
the dataset. By leveraging this sparsity and using sparse vector representations for OHE data, we can reduce storage and computational burdens.
SparseVectors are much more efficient when working with sparse data because they do not store zero values (only store non-zero values and their indices)

We use SparseVector(size, *args) to create a new sparse vector where size is the length of the vector and args is either:
 A list of indices and a list of values( The indices list must be sorted in ascending order)
 For example, SparseVector(5, [1, 3, 4], [10, 30, 40]) will represent the vector [0, 10, 0, 30, 40].
 
 Or we use a key value pair in which key need not be sorted 
 For example, SparseVector(5, [(3, 1), (1, 2)]) will give you the vector [0, 2, 0, 1, 0].
 
'''

# Downloading the data set . The code is from Github

def cleanup_old_downloads():
  from fnmatch import fnmatch

  # Clean up old downloaded files from dbfs:/tmp to prevent QUOTA_EXCEEDED errors.
  for f in dbutils.fs.ls('/tmp'):
    name = str(f.name)
    if fnmatch(name, 'criteo_*'):
      dbutils.fs.rm(str(f.path), recurse=True)

def download_criteo(url):
  from io import BytesIO
  import urllib2
  import tarfile
  import uuid
  import tempfile
  import random
  import string
  import os

  if not url.endswith('dac_sample.tar.gz'):
    raise Exception('Check your download URL. Are you downloading the sample dataset?')

  cleanup_old_downloads()

  # Create a random ID for the directory containing the downloaded file, to avoid any name clashes
  # with any other clusters. (Might not be necessary, but, safety first...)
  rng = random.SystemRandom()
  tlds = ('.org', '.net', '.com', '.info', '.biz')
  random_domain_name = (
    ''.join(rng.choice(string.letters + string.digits) for i in range(64)) +
    rng.choice(tlds)
  )
  random_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, random_domain_name)).replace('-', '_')
  unique_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, random_id)).replace('-', '_')
  dbfs_dir  = 'dbfs:/tmp/criteo_{0}'.format(unique_id)
  dbfs_path = '{0}/data.txt'.format(dbfs_dir)
  dbutils.fs.mkdirs(dbfs_dir)

  # Download the tarball and unpack it.
  tmp = BytesIO()
  req = urllib2.Request(url, headers={'User-Agent': 'Databricks'})
  url_handle = urllib2.urlopen(req)
  tmp.write(url_handle.read())
  tmp.seek(0)
  tf = tarfile.open(fileobj=tmp)
  dac_sample = tf.extractfile('dac_sample.txt')
  dac_sample = '\n'.join([unicode(x.replace('\n', '').replace('\t', ',')) for x in dac_sample])

  # Write the downloaded data to to dbfs:/tmp.
  with tempfile.NamedTemporaryFile(mode='wb', delete=False, prefix='dac', suffix='.txt') as t:
    t.write(dac_sample)
    t.close()
    dbutils.fs.cp('file://{0}'.format(t.name), dbfs_path)
    os.unlink(t.name)

  return dbfs_path
  
criteo_url = 'http://criteolabs.wpengine.com/wp-content/uploads/2015/04/dac_sample.tar.gz'
if ('downloaded_data_file' not in locals()) or (downloaded_data_file is None):
  downloaded_data_file = download_criteo(criteo_url)

if ('raw_df' in locals()) and (raw_df is not None):
  print "raw_df is already loaded. Nothing to do. (Set raw_df=None to reload it, then re-run this cell.)"
else:
  raw_df = sqlContext.read.text(downloaded_data_file).withColumnRenamed("value", "text")

print "raw_df initialized to read from {0}".format(downloaded_data_file)
raw_df.show()

# Splitting data into Train , Validation and Test data set

weights = [.8, .1, .1]
seed = 42

# Use randomSplit with weights and seed
raw_train_df, raw_validation_df, raw_test_df = raw_df.randomSplit(weights, seed)

# Cache and count the DataFrames
n_train = raw_train_df.cache().count()
n_val = raw_validation_df.cache().count()
n_test = raw_test_df.cache().count()
print n_train, n_val, n_test, n_train + n_val + n_test
raw_df.show(10)

# We parse the raw training data to create a DataFrame that we can subsequently use to create an OHE dictionary. 
#we will ignore the first field (which is the 0-1 label), and parse the remaining fields (or raw features). 

def parse_point(point):
    """Converts a comma separated string into a list of (featureID, value) tuples.

    Note:
        featureIDs should start at 0 and increase to the number of features - 1.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.

    Returns:
        list: A list of (featureID, value) tuples.
    """
    parsed_string = point.split(",")
    
    result = [(ind - 1, str(parsed_string[ind])) for ind in xrange(1, len(parsed_string))]
    
    return result

print parse_point(raw_df.select('text').first()[0])

# We create a parse_raw_df function that creates a label column from the first value in the text and a feature column from the rest of the values.

from pyspark.sql.functions import udf, split
from pyspark.sql.types import ArrayType, StructType, StructField, LongType, StringType

parse_point_udf = udf(parse_point, ArrayType(StructType([StructField('_1', LongType()),
                                                         StructField('_2', StringType())])))

def parse_raw_df(raw_df):
    """Convert a DataFrame consisting of rows of comma separated text into labels and feature.


    Args:
        raw_df (DataFrame with a 'text' column): DataFrame containing the raw comma separated data.

    Returns:
        DataFrame: A DataFrame with 'label' and 'feature' columns.
    """
    output_df = (raw_df
                .select(split('text', ",")[0].cast("double").alias("label"),
                       parse_point_udf('text').alias("feature")))
    return output_df.cache()

# Parse the raw training DataFrame
parsed_train_df = parse_raw_df(raw_train_df)

from pyspark.sql.functions import (explode, col)
num_categories = (parsed_train_df
                    .select(explode('feature').alias('feature'))
                    .distinct()
                    .select(col('feature').getField('_1').alias('featureNumber'))
                    .groupBy('featureNumber')
                    .sum()
                    .orderBy('featureNumber')
                    .collect())

print num_categories[2][1]


#Create an OHE dictionary from the dataset

def create_one_hot_dict(input_df):
    """Creates a one-hot-encoder dictionary based on the input data.

    Args:
        input_df (DataFrame with 'features' column): A DataFrame where each row contains a list of
            (featureID, value) tuples.

    Returns:
        dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
            unique integers.
    """
    OHEncoder = (input_df
                .select(explode('features'))
                .distinct()
                .rdd
                .map(lambda r: tuple(r[0]))
                .zipWithIndex()
                .collectAsMap())
    return OHEncoder
	
ctr_ohe_dict =  create_one_hot_dict(parsed_train_df
                                    .select(parsed_train_df.feature
                                    .alias('features')))
num_ctr_ohe_feats = len(ctr_ohe_dict)
print num_ctr_ohe_feats
print ctr_ohe_dict[(0, '')]

# We use this OHE dictionary, by starting with the training data that we've parsed into label and feature columns, to create one-hot-encoded features

from pyspark.sql.functions import udf
from pyspark.mllib.linalg import VectorUDT

def ohe_udf_generator(ohe_dict_broadcast):
    """Generate a UDF that is setup to one-hot-encode rows with the given dictionary.

    Note:
        We'll reuse this function to generate a UDF that can one-hot-encode rows based on a
        one-hot-encoding dictionary built from the training data.  Also, you should calculate
        the number of features before calling the one_hot_encoding function.

    Args:
        ohe_dict_broadcast (Broadcast of dict): Broadcast variable containing a dict that maps
            (featureID, value) to unique integer.

    Returns:
        UserDefinedFunction: A UDF can be used in `DataFrame` `select` statement to call a
            function on each row in a given column.  This UDF should call the one_hot_encoding
            function with the appropriate parameters.
    """
    length = len(ohe_dict_broadcast.value) # or value.keys()
    return udf(lambda x: one_hot_encoding(x, ohe_dict_broadcast, length), VectorUDT())
	
ohe_dict_broadcast = sc.broadcast(ctr_ohe_dict)
ohe_dict_udf = ohe_udf_generator(ohe_dict_broadcast)
ohe_train_df = (parsed_train_df
                  .select('label', ohe_dict_udf(parsed_train_df.feature.alias('features')).alias('features'))
                  .cache())

print ohe_train_df.count()
print ohe_train_df.take(1)

'''

We will now visualize the number of times each of the 233,941 OHE features appears in the training data. 
We first compute the number of times each feature appears, then bucket the features by these counts. 
The buckets are sized by powers of 2, so the first bucket corresponds to features that appear exactly once (2^0), the second to features that appear
twice (2^1) , the third to features that occur between three and four ( 2^2) times, the fifth bucket is five to eight ( 2^3) times and so on. 

The scatter plot below shows the logarithm of the bucket thresholds versus the logarithm of the number of features that have counts that fall in the buckets

Code is from Github
'''

from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import log

get_indices = udf(lambda sv: map(int, sv.indices), ArrayType(IntegerType()))
feature_counts = (ohe_train_df
                   .select(explode(get_indices('features')))
                   .groupBy('col')
                   .count()
                   .withColumn('bucket', log('count').cast('int'))
                   .groupBy('bucket')
                   .count()
                   .orderBy('bucket')
                   .collect())
				   
				   
				   
import matplotlib.pyplot as plt

x, y = zip(*feature_counts)
x, y = x, np.log(y)

def prepare_plot(xticks, yticks, figsize=(10.5, 6), hide_labels=False, grid_color='#999999',
                 grid_width=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hide_labels: axis.set_ticklabels([])
    plt.grid(color=grid_color, linewidth=grid_width, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

# generate layout and plot data
fig, ax = prepare_plot(np.arange(0, 12, 1), np.arange(0, 14, 2))
ax.set_xlabel(r'$\log_e(bucketSize)$'), ax.set_ylabel(r'$\log_e(countInBucket)$')
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
display(fig)


'''
We naturally would like to repeat the process to compute OHE features for the validation and test datasets. 
However, we must be careful, as some categorical values will likely appear in new data that did not exist in the training data.
To deal with this situation, update the one_hot_encoding() function from Part (1d) to ignore previously unseen categories, and then 
compute OHE features for the validation data

'''

def one_hot_encoding(raw_feats, ohe_dict_broadcast, num_ohe_feats):
    """Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        You should ensure that the indices used to create a SparseVector are sorted, and that the
        function handles missing features.

    Args:
        raw_feats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sample_one)
        ohe_dict_broadcast (Broadcast of dict): Broadcast variable containing a dict that maps
            (featureID, value) to unique integer.
        num_ohe_feats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length num_ohe_feats with indices equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.
    """
    return SparseVector(num_ohe_feats, [(ohe_dict_broadcast.value[feat_tuple], 1.) for feat_tuple in raw_feats \
                                        if ohe_dict_broadcast.value.has_key(feat_tuple)])
                                       

ohe_dict_missing_udf = ohe_udf_generator(ohe_dict_broadcast)
parsed_validation_df = parse_raw_df(raw_validation_df.select('text'))
ohe_validation_df = (parsed_validation_df
                  .select('label', ohe_dict_missing_udf(parsed_validation_df.feature.alias('features')).alias('features'))
                  .cache())

ohe_validation_df.count()
ohe_validation_df.show(1, truncate=False)

'''
We are ready to train our first CTR classifier using Logistic regression

We use LogisticRegression from the pyspark.ml package to train a model using ohe_train_df with the given hyperparameter configuration.
LogisticRegression.fit returns a LogisticRegressionModel. Next, we'll use the LogisticRegressionModel.coefficients and 
LogisticRegressionModel.intercept attributes to print out some details of the model's parameters.

'''

standardization = False
elastic_net_param = 0.0
reg_param = .01
max_iter = 20

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=max_iter, regParam=reg_param, elasticNetParam=elastic_net_param, standardization=standardization)

lr_model_basic = lr.fit(ohe_train_df)

print 'intercept: {0}'.format(lr_model_basic.intercept)
print 'length of coefficients: {0}'.format(len(lr_model_basic.coefficients))
sorted_coefficients = sorted(lr_model_basic.coefficients)[:5]

# Function to calculate log loss

from pyspark.sql.functions import when, log, col
epsilon = 1e-16

def add_log_loss(df):
    """Computes and adds a 'log_loss' column to a DataFrame using 'p' and 'label' columns.

    Note:
        log(0) is undefined, so when p is 0 we add a small value (epsilon) to it and when
        p is 1 we subtract a small value (epsilon) from it.

    Args:
        df (DataFrame with 'p' and 'label' columns): A DataFrame with a probability column
            'p' and a 'label' column that corresponds to y in the log loss formula.

    Returns:
        DataFrame: A new DataFrame with an additional column called 'log_loss' where 'log_loss' column contains the loss value as explained above.
    """
    new_df = (df
             .select("*", when(df.label == 1, 0. - log(df.p + epsilon)).\
                          otherwise(0. - log(1. - df.p + epsilon))
                    .alias('log_loss'))
             )
    return new_df
	

'''
Baseline model where we predict the same for all points with etting the predicted value equal to the fraction 
of training points that correspond to click-through events (i.e., where the label is one)
'''

from pyspark.sql.functions import lit
class_one_frac_train = (ohe_train_df
                       .groupBy()
                       .mean('label')
                       .collect())[0][0]
print 'Training class one fraction = {0:.3f}'.format(class_one_frac_train)

# CTR is 20 % which seems very high

log_loss_tr_base = (add_log_loss(ohe_train_df
                                .select("*", lit(class_one_frac_train)
                                       .alias('p')))
                                .groupBy()
                                .mean('log_loss')
                                .collect())[0][0]

print 'Baseline Train Logloss = {0:.3f}\n'.format(log_loss_tr_base)

# Function to calculate probabilities using sigmoid function

from pyspark.sql.types import DoubleType
from math import exp #  exp(-t) = e^-t

def add_probability(df, model):
    """Adds a probability column ('p') to a DataFrame given a model"""
    coefficients_broadcast = sc.broadcast(model.coefficients)
    intercept = model.intercept

    def get_p(features):
        """Calculate the probability for an observation given a list of features.

        Note:
            We'll bound our raw prediction between 20 and -20 for numerical purposes.

        Args:
            features: the features

        Returns:
            float: A probability between 0 and 1.
        """
        # Compute the raw value
        raw_prediction = intercept + coefficients_broadcast.value.dot(features)
        # Bound the raw value between 20 and -20
        raw_prediction = min(20, max(-20, raw_prediction))
        # Return the probability
        probability = 1.0 / (1 + exp(-raw_prediction)) # sigmoid
        return probability

    get_p_udf = udf(get_p, DoubleType())
    return df.withColumn('p', get_p_udf('features'))

add_probability_model_basic = lambda df: add_probability(df, lr_model_basic)
training_predictions = add_probability_model_basic(ohe_train_df).cache()

training_predictions.show(5)

# Evaluation using mean log loss 

def evaluate_results(df, model, baseline=None):
    """Calculates the log loss for the data given the model.

    Note:
        If baseline has a value the probability should be set to baseline before
        the log loss is calculated.  Otherwise, use add_probability to add the
        appropriate probabilities to the DataFrame.

    Args:
        df (DataFrame with 'label' and 'features' columns): A DataFrame containing
            labels and features.
        model (LogisticRegressionModel): A trained logistic regression model. This
            can be None if baseline is set.
        baseline (float): A baseline probability to use for the log loss calculation.

    Returns:
        float: Log loss for the data.
    """
    if model != None:
      with_probability_df = add_probability(df, model)
    else:
      with_probability_df = df.withColumn('p', lit(baseline))
      
    with_log_loss_df = add_log_loss(with_probability_df)
    
    log_loss = (with_log_loss_df
               .groupBy()
               .mean('log_loss')
               .collect()[0][0]
               )
    
    return log_loss

log_loss_train_model_basic = evaluate_results(ohe_train_df, lr_model_basic)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(log_loss_tr_base, log_loss_train_model_basic))
	   
# compute the validation log loss

log_loss_val_base = evaluate_results(ohe_validation_df, None, class_one_frac_train)

log_loss_val_l_r0 = evaluate_results(ohe_validation_df, lr_model_basic)
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(log_loss_val_base, log_loss_val_l_r0))

#ROC curve

labels_and_scores = add_probability_model_basic(ohe_validation_df).select('label', 'p')
labels_and_weights = labels_and_scores.collect()
labels_and_weights.sort(key=lambda (k, v): v, reverse=True)
labels_by_weight = np.array([k for (k, v) in labels_and_weights])

length = labels_by_weight.size
true_positives = labels_by_weight.cumsum()
num_positive = true_positives[-1]
false_positives = np.arange(1.0, length + 1, 1.) - true_positives

true_positive_rate = true_positives / num_positive
false_positive_rate = false_positives / (length - num_positive)

# Generate layout and plot data
fig, ax = prepare_plot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(false_positive_rate, true_positive_rate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
display(fig)

#  To reduce the dimensionality of the feature space, we will use feature hashing. Code is from Github

from collections import defaultdict
import hashlib

def hash_function(raw_feats, num_buckets, print_mapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use print_mapping=True for debug purposes and to better understand how the hashing works.

    Args:
        raw_feats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        num_buckets (int): Number of buckets to use as features.
        print_mapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = { category + ':' + str(ind):
                int(int(hashlib.md5(category + ':' + str(ind)).hexdigest(), 16) % num_buckets)
                for ind, category in raw_feats}
    if(print_mapping): print mapping

    def map_update(l, r):
        l[r] += 1.0
        return l

    sparse_features = reduce(map_update, mapping.values(), defaultdict(float))
    return dict(sparse_features)

	
from pyspark.mllib.linalg import Vectors
num_hash_buckets = 2 ** 15

# UDF that returns a vector of hashed features given an Array of tuples
tuples_to_hash_features_udf = udf(lambda x: Vectors.sparse(num_hash_buckets, hash_function(x, num_hash_buckets)), VectorUDT())

def add_hashed_features(df):
    """Return a DataFrame with labels and hashed features.

    Note:
        Make sure to cache the DataFrame that you are returning.

    Args:
        df (DataFrame with 'tuples' column): A DataFrame containing the tuples to be hashed.

    Returns:
        DataFrame: A DataFrame with a 'label' column and a 'features' column that contains a
            SparseVector of hashed features.
    """
    return df.select('label', tuples_to_hash_features_udf('feature').alias('features'))


hash_train_df = add_hashed_features(parsed_train_df)
hash_validation_df = add_hashed_features(parsed_validation_df)
hash_test_df = add_hashed_features(parse_raw_df(raw_test_df.select('text')))

hash_train_df.show()

#  Logistic model with hashed features

standardization = False
elastic_net_param = 0.7
reg_param = .001
max_iter = 20

lr_hash = LogisticRegression(maxIter = max_iter, regParam = reg_param, elasticNetParam = elastic_net_param, standardization = standardization)

lr_model_hashed = lr_hash.fit(hash_train_df)
print 'intercept: {0}'.format(lr_model_hashed.intercept)
print len(lr_model_hashed.coefficients)

log_loss_train_model_hashed = evaluate_results(hash_train_df, lr_model_hashed)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\thashed = {1:.3f}'
       .format(log_loss_tr_base, log_loss_train_model_hashed))
	   
#  Evaluate on the test set

log_loss_test = evaluate_results(hash_test_df, lr_model_hashed)

# Log loss for the baseline model
class_one_frac_test = class_one_frac_train
print 'Class one fraction for test data: {0}'.format(class_one_frac_test)
log_loss_test_baseline = evaluate_results(hash_test_df, None, class_one_frac_test)

print ('Hashed Features Test Log Loss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(log_loss_test_baseline, log_loss_test))

