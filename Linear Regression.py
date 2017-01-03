'''
Linear regression on the subset million song data set 
The data is a public data set available on http://labrosa.ee.columbia.edu/millionsong/
Goal is to predict the release year of a song given a set of audio features.
We build 2 models using Gradient descent and Spark ML and tune hyper parameters using grid search
'''	
# load testing library

import os.path
file_name = os.path.join('databricks-datasets', 'cs190', 'data-001', 'millionsong.txt')

raw_data_df = sqlContext.read.load(file_name, 'text')
sample_points = raw_data_df.take(5)
print sample_points #Print top 5 data points

'''
In MLlib, labeled training instances are stored using the LabeledPoint object.
We write the parse_points function that takes, as input, a DataFrame of comma-separated strings. We'll pass it the raw_data_df DataFrame.
It should parse each row in the DataFrame into individual elements, using Spark's select and split methods.
Eg :  '2001.0,0.884,0.610,0.600,0.474,0.247,0.357,0.344,0.33,0.600,0.425,0.60,0.419'
In this raw data point, 2001.0 is the label, and the remaining values are features
'''

from pyspark.mllib.regression import LabeledPoint
import numpy as np

# Here is a sample raw data point:
# '2001.0,0.884,0.610,0.600,0.474,0.247,0.357,0.344,0.33,0.600,0.425,0.60,0.419'
# In this raw data point, 2001.0 is the label, and the remaining values are features

from pyspark.sql import functions as sql_functions

def parse_points(df):
    """Converts a DataFrame of comma separated unicode strings into a DataFrame of `LabeledPoints`.

    Args:
        df: DataFrame where each row is a comma separated unicode string. The first element in the string
            is the label and the remaining elements are the features.

    Returns:
        DataFrame: Each row is converted into a `LabeledPoint`, which consists of a label and
            features. To convert an RDD to a DataFrame, simply call toDF().
    """
    # Split:
    
    df = df.select(sql_functions.split(df['value'], ','))
    
    # Return label, features: each row is a list
    
    df_labelpoint = df.map(lambda rowlist: LabeledPoint(float(rowlist[0][0]), [float(x) for x in rowlist[0][1:]])).toDF()
    
    return df_labelpoint

parsed_points_df = parse_points(raw_data_df)
first_point_features = parsed_points_df.first().features # LabelPoint.features
first_point_label = parsed_points_df.first().label # LabelPoint.label
print first_point_features, first_point_label

d = len(first_point_features)
print d

'''
We will look at the raw features for 50 data points by generating a heatmap that visualizes each feature on a grey-scale and shows the variation
of each feature across the 50 sample data points. The features are all between 0 and 1, with values closer to 1 represented via darker shades of grey.

'''

# Code taken from Stack overflow

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# takeSample(withReplacement, num, [seed]) randomly selects num elements from the dataset with/without replacement, and has an
# optional seed parameter that one can set for reproducible results

data_values = (parsed_points_df
               .rdd
               .map(lambda lp: lp.features.toArray())
               .takeSample(False, 50, 47))

# You can uncomment the line below to see randomly selected features.  These will be randomly
# selected each time you run the cell because there is no set seed.  Note that you should run
# this cell with the line commented out when answering the lab quiz questions.
# data_values = (parsedPointsDF
#                .rdd
#                .map(lambda lp: lp.features.toArray())
#                .takeSample(False, 50))

def prepare_plot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                 gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

# generate layout and plot
fig, ax = prepare_plot(np.arange(.5, 11, 1), np.arange(.5, 49, 1), figsize=(8,7), hideLabels=True,
                       gridColor='#eeeeee', gridWidth=1.1)
image = plt.imshow(data_values,interpolation='nearest', aspect='auto', cmap=cm.Greys)
for x, y, s in zip(np.arange(-.125, 12, 1), np.repeat(-.75, 12), [str(x) for x in range(12)]):
    plt.text(x, y, s, color='#999999', size='10')
plt.text(4.7, -3, 'Feature', color='#999999', size='11'), ax.set_ylabel('Observation')
display(fig)


content_stats = (parsed_points_df
                 .selectExpr("min(label)", "max(label)")).collect()

min_year = content_stats[0][0] # Finding maximum year in the dataframe usng Max function
max_year = content_stats[0][1] # Finding minimum year in the dataframe usng Min function

print min_year, max_year


'''
As the years are in 1900s and 2000s we are creating a new DataFrame in which the labels are shifted such that smallest label equals zero
After, we use withColumnRenamed to rename the appropriate columns to features and label.
'''
parsed_data_df = parsed_points_df.map(lambda row: LabeledPoint(row['label'] - min_year, row['features'])).toDF()

# View the first point
print '\n{0}'.format(parsed_data_df.first())

'''
We will look at the labels before and after shifting them. Both scatter plots below visualize tuples storing:
a label value and the number of training points with this label.
The first scatter plot uses the initial labels, while the second one uses the shifted labels.
Note that the two plots look the same except for the labels on the x-axis.

Code was taken from Stack overflow and Github
'''

# get data for plot
old_data = (parsed_points_df
             .rdd
             .map(lambda lp: (lp.label, 1))
             .reduceByKey(lambda x, y: x + y)
             .collect())
x, y = zip(*old_data)

# generate layout and plot data
fig, ax = prepare_plot(np.arange(1920, 2050, 20), np.arange(0, 150, 20))
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
ax.set_xlabel('Year'), ax.set_ylabel('Count')
display(fig)

# get data for plot
new_data = (parsed_points_df
             .rdd
             .map(lambda lp: (lp.label, 1))
             .reduceByKey(lambda x, y: x + y)
             .collect())
x, y = zip(*new_data)

# generate layout and plot data
fig, ax = prepare_plot(np.arange(0, 120, 20), np.arange(0, 120, 20))
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
ax.set_xlabel('Year (shifted)'), ax.set_ylabel('Count')
display(fig)
pass


# Splitting into Training, validation, and test sets

weights = [.8, .1, .1]
seed = 42
parsed_train_data_df, parsed_val_data_df, parsed_test_data_df = parsed_data_df.randomSplit(weights, seed)
parsed_train_data_df.cache()
parsed_val_data_df.cache()
parsed_test_data_df.cache()
n_train = parsed_train_data_df.count()
n_val = parsed_val_data_df.count()
n_test = parsed_test_data_df.count()

print n_train, n_val, n_test, n_train + n_val + n_test
print parsed_data_df.count()

#Create and evaluate a baseline model

average_train_year = (parsed_train_data_df
                        .selectExpr("avg(label)")
                        .first())[0]
print average_train_year # Getting average year. We use it to predict the average for all labels

'''
To check performance of this naive baseline we will use root mean squared error (RMSE). Using Regression Evaluator, compute the RMSE given
a dataset of (prediction, label) tuples.

'''

from pyspark.ml.evaluation import RegressionEvaluator

def calc_RMSE(dataset):
 '''Calculates the root mean squared error for an dataset of (prediction, label) tuples.

    Args:
        dataset (DataFrame of (float, float)): A `DataFrame` consisting of (prediction, label) tuples.

    Returns:
        float: The square root of the mean of the squared errors.
    '''
	
  return evaluator.evaluate(dataset)
  
 # Finding RMSE for train , validation and test data sets using the calc_RMSE function
 
preds_and_labels_train = parsed_train_data_df.map(lambda x: (average_train_year, x.label))
preds_and_labels_train_df = sqlContext.createDataFrame(preds_and_labels_train, ["prediction", "label"])
rmse_train_base = calc_RMSE(preds_and_labels_train_df)

preds_and_labels_val = parsed_val_data_df.map(lambda x: (average_train_year, x.label))
preds_and_labels_val_df = sqlContext.createDataFrame(preds_and_labels_val, ["prediction", "label"])
rmse_val_base = calc_RMSE(preds_and_labels_val_df)

preds_and_labels_test = parsed_test_data_df.map(lambda x: (average_train_year, x.label))
preds_and_labels_test_df = sqlContext.createDataFrame(preds_and_labels_test, ["prediction", "label"])
rmse_test_base = calc_RMSE(preds_and_labels_test_df)

print 'Baseline Train RMSE = {0:.3f}'.format(rmse_train_base)
print 'Baseline Validation RMSE = {0:.3f}'.format(rmse_val_base)
print 'Baseline Test RMSE = {0:.3f}'.format(rmse_test_base)

'''
We will visualize predictions on the validation dataset. The scatter plots below visualize tuples storing i) the predicted value and ii) true label. 
The first scatter plot represents the ideal situation where the predicted value exactly equals the true label, while the second plot uses the baseline 
predictor (i.e., average_train_year) for all predicted values. Further note that the points in the scatter plots are color-coded, ranging from 
light yellow when the true and predicted values are equal to bright red when they drastically differ.

'''

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
cmap = get_cmap('YlOrRd')
norm = Normalize()

def squared_error(label, prediction):
    """Calculates the squared error for a single prediction."""
    return float((label - prediction)**2)

actual = np.asarray(parsed_val_data_df
                    .select('label')
                    .collect())
error = np.asarray(parsed_val_data_df
                   .rdd
                   .map(lambda lp: (lp.label, lp.label))
                   .map(lambda (l, p): squared_error(l, p))
                   .collect())
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = prepare_plot(np.arange(0, 100, 20), np.arange(0, 100, 20))
plt.scatter(actual, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.5) # Note that we are compating actual with Actual for ideal plot
ax.set_xlabel('Predicted'), ax.set_ylabel('Actual')
display(fig) # The first scatter plot represents the ideal situation where the predicted value exactly equals the true label

def squared_error(label, prediction):
    """Calculates the squared error for a single prediction."""
    return float((label - prediction)**2)

predictions = np.asarray(parsed_val_data_df
                         .rdd
                         .map(lambda lp: average_train_year)
                         .collect())
error = np.asarray(parsed_val_data_df
                   .rdd
                   .map(lambda lp: (lp.label, average_train_year))
                   .map(lambda (l, p): squared_error(l, p))
                   .collect())
norm = Normalize()
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = prepare_plot(np.arange(53.0, 55.0, 0.5), np.arange(0, 100, 20))
ax.set_xlim(53, 55)
plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.3) #Note that we are compating prediction with Avg label
ax.set_xlabel('Predicted'), ax.set_ylabel('Actual')
display(fig) # the second plot uses the baseline predictor (i.e., average_train_year) for all predicted values



'''
We now create a model via gradient descent (we'll omit the intercept for now).
w (​i+1) = w(i) - α ∑(W(i)T  X(j) - y(i))X(j)
We use the DenseVector dot method.

'''

from pyspark.mllib.linalg import DenseVector

def gradient_summand(weights, lp):
    """Calculates the gradient summand for a given weight and `LabeledPoint`.

    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        weights (DenseVector): An array of model weights (betas).
        lp (LabeledPoint): The `LabeledPoint` for a single observation.

    Returns:
        DenseVector: An array of values the same length as `weights`.  The gradient summand.
    """
    return (weights.dot(lp.features) - lp.label) * lp.features

def get_labeled_prediction(weights, observation):
    """Calculates predictions and returns a (prediction, label) tuple.

    Note:
        The labels should remain unchanged as we'll use this information to calculate prediction
        error later.

    Args:
        weights (np.ndarray): An array with one weight for each features in `trainData`.
        observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
            features for the data point.

    Returns:
        tuple: A (prediction, label) tuple. Convert the return type of the label and prediction to a float.
    """
    return (float(weights.dot(observation.features)), float(observation.label))

	
def linreg_gradient_descent(train_data, num_iters):
    """Calculates the weights and error for a linear regression model trained with gradient descent.

    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        train_data (RDD of LabeledPoint): The labeled data for use in training the model.
        num_iters (int): The number of iterations of gradient descent to perform.

    Returns:
        (np.ndarray, np.ndarray): A tuple of (weights, training errors).  Weights will be the
            final weights (one weight per feature) for the model, and training errors will contain
            an error (RMSE) for each iteration of the algorithm.
    """
    # The length of the training data
    n = train_data.count()
    # The number of features in the training data
    d = len(train_data.first().features)
    w = np.zeros(d)
    alpha = 1.0
    # We will compute and store the training error after each iteration
    error_train = np.zeros(num_iters)
    for i in range(num_iters):
        # Use get_labeled_prediction with trainData to obtain an RDD of (label, prediction)
        # tuples.  Note that the weights all equal 0 for the first iteration, so the predictions will
        # have large errors to start.
        
		preds_and_labels_train = train_data.map(lambda observation: get_labeled_prediction(w, observation))
        preds_and_labels_train_df = sqlContext.createDataFrame(preds_and_labels_train, ["prediction", "label"])
        error_train[i] = calc_RMSE(preds_and_labels_train_df)

        # Calculate the `gradient`.  Make use of the `gradient_summand` function
        # Note that `gradient` should be a `DenseVector` of length `d`.
        
		gradient = (train_data.map(lambda observation: gradient_summand(w, observation))
                              .reduce(lambda obs1, obs2: obs1+obs2))

        # Update the weights
        alpha_i = alpha / (n * np.sqrt(i+1))
        w -= alpha_i * gradient # the GD update rule
    return w, error_train
	
'''
We train a linear regression model on all of our training data and evaluate its accuracy on the validation set.
We use the functions created before
'''

num_iters = 50 # Using a small value as we have limited computation power
weights_LR0, error_train_LR0 = linreg_gradient_descent(parsed_train_data_df, num_iters)

preds_and_labels = (parsed_val_data_df
                      .map(lambda lp: get_labeled_prediction(weights_LR0, lp)))
preds_and_labels_df = sqlContext.createDataFrame(preds_and_labels, ["prediction", "label"])
rmse_val_LR0 = calc_RMSE(preds_and_labels_df)

print 'Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}'.format(rmse_val_base,
                                                                       rmse_val_LR0)



# We look at the log of the training error as a function of iteration. 


#The first scatter plot visualizes the logarithm of the training error for all 50 iterations.  
norm = Normalize()
clrs = cmap(np.asarray(norm(np.log(error_train_LR0))))[:,0:3]
fig, ax = prepare_plot(np.arange(0, 60, 10), np.arange(2, 6, 1))
ax.set_ylim(2, 6)
plt.scatter(range(0, num_iters), np.log(error_train_LR0), s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
ax.set_xlabel('Iteration'), ax.set_ylabel(r'$\log_e(errorTrainLR0)$')
display(fig)


#The second plot shows the training error itself, focusing on the final 44 iterations
norm = Normalize()
clrs = cmap(np.asarray(norm(error_train_LR0[6:])))[:,0:3]

fig, ax = prepare_plot(np.arange(0, 60, 10), np.arange(17, 22, 1))
ax.set_ylim(17.8, 21.2)
plt.scatter(range(0, num_iters-6), error_train_LR0[6:], s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
ax.set_xticklabels(map(str, range(6, 66, 10)))
ax.set_xlabel('Iteration'), ax.set_ylabel(r'Training Error')
display(fig)

'''
we can do better by adding an intercept, using regularization, and (based on the previous visualization) training for more iterations.
First use LinearRegression to train a model with elastic net regularization and an intercept. This method returns a LinearRegressionModel. 
Next, use the model's coefficients (weights) and intercept attributes to print out the model's parameters.
'''

from pyspark.ml.regression import LinearRegression
# Values to use when training the linear regression model

num_iters = 500  # iterations
reg = 1e-1  # regParam
alpha = .2  # elasticNetParam
use_intercept = True  # intercept

lin_reg = LinearRegression(maxIter = num_iters, regParam=reg, elasticNetParam=alpha)
first_model = lin_reg.fit(parsed_train_data_df)

# coeffsLR1 stores the model coefficients; interceptLR1 stores the model intercept
#coeffs_LR1 = first_model.weights
coeffs_LR1 = first_model.coefficients
intercept_LR1 = first_model.intercept
print coeffs_LR1, intercept_LR1


# We use the LinearRegressionModel.transform() method to make predictions on the parsed_train_data_df.
sample_prediction = first_model.transform(parsed_train_data_df)
display(sample_prediction)

# Evaluate RMSE using 	calc_RMSE()
val_pred_df = first_model.transform(parsed_val_data_df)
rmse_val_LR1 = calc_RMSE(val_pred_df.select(val_pred_df['prediction'], val_pred_df['label']))

print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}' +
       '\n\tLR1 = {2:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1)
	   
# We Perform grid search to find a good regularization parameter between (1e-10, 1e-5, and 1.0 )

best_RMSE = rmse_val_LR1
best_reg_param = reg
best_model = first_model

num_iters = 500  # iterations
alpha = .2  # elasticNetParam
use_intercept = True  # intercept

for reg in [1e-10, 1e-5, 1.0]:
    lin_reg = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
    model = lin_reg.fit(parsed_train_data_df)
    val_pred_df = model.transform(parsed_val_data_df)

    rmse_val_grid = calc_RMSE(val_pred_df)
    print rmse_val_grid

    if rmse_val_grid < best_RMSE:
        best_RMSE = rmse_val_grid
        best_reg_param = reg
        best_model = model

rmse_val_LR_grid = best_RMSE

print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n' +
       '\tLRGrid = {3:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1, rmse_val_LR_grid)
	   
	   
# Visualization : Predicted vs. actual

parsed_val_df = best_model.transform(parsed_val_data_df)
predictions = np.asarray(parsed_val_df
                         .select('prediction')
                         .collect())
actual = np.asarray(parsed_val_df
                      .select('label')
                      .collect())
error = np.asarray(parsed_val_df
                     .rdd
                     .map(lambda lp: squared_error(lp.label, lp.prediction))
                     .collect())

norm = Normalize()
clrs = cmap(np.asarray(norm(error)))[:,0:3]

fig, ax = prepare_plot(np.arange(0, 120, 20), np.arange(0, 120, 20))
ax.set_xlim(15, 82), ax.set_ylim(-5, 105)
plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=.5)
ax.set_xlabel('Predicted'), ax.set_ylabel(r'Actual')
display(fig)


# visualization of hyperparameter search using a larger set of hyperparameters

from matplotlib.colors import LinearSegmentedColormap

# Saved parameters and results, to save the time required to run 36 models
num_iters = 500
reg_params = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
alpha_params = [0.0, .1, .2, .4, .8, 1.0]
rmse_val = np.array([[ 15.317156766552452, 15.327211561989827, 15.357152971253697, 15.455092206273847, 15.73774335576239,
                       16.36423857334287, 15.315019185101972, 15.305949211619886, 15.355590337955194, 15.573049001631558,
                       16.231992712117222, 17.700179790697746, 15.305266383061921, 15.301104931027034, 15.400125020566225,
                       15.824676190630191, 17.045905140628836, 19.365558346037535, 15.292810983243772, 15.333756681057828,
                       15.620051033979871, 16.631757941340428, 18.948786862836954, 20.91796910560631, 15.308301384150049,
                       15.522394576046239, 16.414106221093316, 18.655978799189178, 20.91796910560631, 20.91796910560631,
                       15.33442896030322, 15.680134490745722, 16.86502909075323, 19.72915603626022, 20.91796910560631,
                       20.91796910560631 ]])

num_rows, num_cols = len(alpha_params), len(reg_params)
rmse_val = np.array(rmse_val)
rmse_val.shape = (num_rows, num_cols)

fig, ax = prepare_plot(np.arange(0, num_cols, 1), np.arange(0, num_rows, 1), figsize=(8, 7), hideLabels=True,
                       gridWidth=0.)
ax.set_xticklabels(reg_params), ax.set_yticklabels(alpha_params)
ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Alpha')

colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
image = plt.imshow(rmse_val,interpolation='nearest', aspect='auto',
                    cmap = colors)
display(fig)

# Zoom into the top left
alpha_params_zoom, reg_params_zoom = alpha_params[1:5], reg_params[:4]
rmse_val_zoom = rmse_val[1:5, :4]

num_rows, num_cols = len(alpha_params_zoom), len(reg_params_zoom)

fig, ax = prepare_plot(np.arange(0, num_cols, 1), np.arange(0, num_rows, 1), figsize=(8, 7), hideLabels=True,
                       gridWidth=0.)
ax.set_xticklabels(reg_params_zoom), ax.set_yticklabels(alpha_params_zoom)
ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Alpha')

colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
image = plt.imshow(rmse_val_zoom, interpolation='nearest', aspect='auto',
                    cmap = colors)
display(fig)


#we will add features that capture the two-way interactions between our existing features

import itertools

def two_way_interactions(lp):
    """Creates a new `LabeledPoint` that includes two-way interactions.

    Note:
        For features [x, y] the two-way interactions would be [x^2, x*y, y*x, y^2] and these
        would be appended to the original [x, y] feature list.

    Args:
        lp (LabeledPoint): The label and features for this observation.

    Returns:
        LabeledPoint: The new `LabeledPoint` should have the same label as `lp`.  Its features
            should include the features from `lp` followed by the two-way interaction features.
    """
    new_features = [x * y for x, y in itertools.product(lp.features, lp.features)]
    interaction_features = np.hstack((lp.features, new_features))
    
    return LabeledPoint(lp.label, interaction_features)

print two_way_interactions(LabeledPoint(0.0, [2, 3]))

# Transform the existing train, validation, and test sets to include two-way interactions.
# Remember to convert them back to DataFrames at the end.
train_data_interact_df = parsed_train_data_df.map(lambda lp: two_way_interactions(lp)).toDF() 
val_data_interact_df = parsed_val_data_df.map(lambda lp: two_way_interactions(lp)).toDF()
test_data_interact_df = parsed_test_data_df.map(lambda lp: two_way_interactions(lp)).toDF()

num_iters = 500
reg = 1e-10
alpha = .2
use_intercept = True

#Build interaction model
lin_reg = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
model_interact = lin_reg.fit(train_data_interact_df)
preds_and_labels_interact_df = model_interact.transform(val_data_interact_df)
rmse_val_interact = calc_RMSE(preds_and_labels_interact_df)

print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n\tLRGrid = ' +
       '{3:.3f}\n\tLRInteract = {4:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1,
                                                 rmse_val_LR_grid, rmse_val_interact)
												 
# Evaluate interaction model on test data which we are using for the first time
preds_and_labels_test_df = model_interact.transform(test_data_interact_df)
rmse_test_interact = calc_RMSE(preds_and_labels_test_df)

print ('Test RMSE:\n\tBaseline = {0:.3f}\n\tLRInteract = {1:.3f}'
       .format(rmse_test_base, rmse_test_interact))
	   
# We now create the interaction model using a Pipeline using  the PolynomialExpansion transformer

from pyspark.ml import Pipeline
from pyspark.ml.feature import PolynomialExpansion

num_iters = 500
reg = 1e-10
alpha = .2
use_intercept = True

polynomial_expansion =PolynomialExpansion(degree = 2, inputCol = "features", outputCol = "polyFeatures")
linear_regression = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha,
                                     fitIntercept=use_intercept, featuresCol='polyFeatures')

pipeline = Pipeline(stages=[polynomial_expansion, linear_regression])
pipeline_model = pipeline.fit(parsed_train_data_df)

predictions_df = pipeline_model.transform(parsed_test_data_df)

evaluator = RegressionEvaluator()
rmse_test_pipeline = evaluator.evaluate(predictions_df, {evaluator.metricName: "rmse"})
print('RMSE for test data set using pipelines: {0:.3f}'.format(rmse_test_pipeline))



