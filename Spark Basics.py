#data is just a normal Python list, containing Python tuples objects
data[0]
len(data) #Length 10000
#Create a dataframe using Sql Context
#DataFrames are ultimately represented as RDDs, with additional meta-data.
dataDF = sqlContext.createDataFrame(data, ('last_name', 'first_name', 'ssn', 'occupation', 'age'))

#Print type
print 'type of dataDF: {0}'.format(type(dataDF))
dataDF.printSchema() #DataFrame's schema 

sqlContext.registerDataFrameAsTable(dataDF, 'dataframe') # register the newly created DataFrame as a named table
dataDF.rdd.getNumPartitions() # Prints partitions DataFrame to be split into
newDF = dataDF.distinct().select('*')
newDF.explain(True) #examine the query plan using the explain() function

# Transform dataDF through a select transformation and rename the newly created '(age -1)' column to 'age'
# Because select is a transformation and Spark uses lazy evaluation, no jobs, stages,
# or tasks will be launched when we run this code.
subDF = dataDF.select('last_name', 'first_name', 'ssn', 'occupation', (dataDF.age - 1).alias('age'))

# Let's collect the data
results = subDF.collect()
subDF.show()
subDF.show(n=30, truncate=False) # View 30 rows
display(subDF) #Databricks helper function to get nicer views
print subDF.count() # prints Counts of elements

#Filter Transformation
filteredDF = dataDF.filter(dataDF.age < 10)
filteredDF.show(truncate=False)
filteredDF.count()

#Using UDF with Lambda functions
from pyspark.sql.types import BooleanType
less_ten = udf(lambda s: s < 10 , BooleanType()) # Filtering numbers less than 10
lambdaDF = subDF.filter(less_ten(subDF.age))
lambdaDF.show()
lambdaDF.count()

#Let's collect the even values less than 10
even = udf(lambda s: s % 2 == 0, BooleanType())
evenDF = lambdaDF.filter(even(lambdaDF.age))
evenDF.show()
evenDF.count()


display(filteredDF.take(4)) # return the first 4 elements of the DataFrame.

# Get the five oldest people in the list. To do that, sort by age in descending order.
display(dataDF.orderBy(dataDF.age.desc()).take(5)) #Sort by age in descending order

# Something like orderBy('age'.desc()) would not work, because there's no desc() method on Python string objects
#That's why we needed the column expression

display(dataDF.orderBy('age').take(5)) # Clubbing the expressions

print dataDF.distinct().count() # distinct() filters out duplicate rows, and it considers all columns.

print dataDF.dropDuplicates(['first_name', 'last_name']).count()
#dropDuplicates() is like distinct(), except that it allows us to specify the columns to compare. 


#drop() is like the opposite of select() it drops a specifed column from a DataFrame
dataDF.drop('occupation').drop('age').show()

# groupBy() is a transformations. It allows you to perform aggregations on a DataFrame.
# count() is the common aggregation, but there are others (like sum(), max(), and avg()

dataDF.groupBy('occupation').count().show(truncate=False) # Count of Column
dataDF.groupBy().avg('age').show(truncate=False) # Average of column
print "Maximum age: {0}".format(dataDF.groupBy().max('age').first()[0]) # Max 
print "Minimum age: {0}".format(dataDF.groupBy().min('age').first()[0]) # Minimum


'''
sample() transformation returns a new DataFrame with a random sample of elements from the dataset
withReplacement argument specifies if it is okay to randomly pick the same item multiple times from the parent DataFrame
withReplacement=True, you can get the same item back multiple times 
It takes in a fraction parameter, which specifies the fraction elements in the dataset you want to return.
So a fraction value of 0.20 returns 20% of the elements in the DataFrame.
It also takes an optional seed parameter that allows you to specify a seed value for the random number generator
'''

sampledDF = dataDF.sample(withReplacement=False, fraction=0.10)
print sampledDF.count()
sampledDF.show()

print dataDF.sample(withReplacement=False, fraction=0.05).count()

#if you plan to use a DataFrame more than once, then you should tell Spark to cache it

# Cache the DataFrame
filteredDF.cache()
# Trigger an action
print filteredDF.count()
# Check if it is cached
print filteredDF.is_cached
# If we are done with the DataFrame we can unpersist it so that its memory can be reclaimed
filteredDF.unpersist()
# Check if it is cached
print filteredDF.is_cached

''' Recommended Spark coding style '''
df2 = df1.transformation1()
df2.action1()
df3 = df2.transformation2()
df3.action2()
''' Expert Style of coding
Make use of Lambda functions
To make the expert coding style more readable, enclose the statement in parentheses and put each method, transformation, or action on a separate line.
'''
df.transformation1().transformation2().action()

# Cleaner code through lambda use
myUDF = udf(lambda v: v < 10)
subDF.filter(myUDF(subDF.age) == True)

# Final version
from pyspark.sql.functions import *
(dataDF
 .filter(dataDF.age > 20)
 .select(concat(dataDF.first_name, lit(' '), dataDF.last_name), dataDF.occupation)
 .show(truncate=False)
 )
 
 # Multiply the elements in dataset by five, keep just the even values, and sum those values
finalSum = (dataset
           .map(lambda x: x*5)
           .filter(lambda x: (x%2==0))
           .reduce(lambda x,y: x+y)
           )
print finalSum

#NUMPY AND MATH BASICS

#Scalar multiplication
import numpy as np
simpleArray = np.array([1, 2, 3])
timesFive = 5*simpleArray

#Element-wise multiplication and dot product
u = np.arange(0, 5, .5)
v = np.arange(5, 10, .5)
elementWise = u*v
dotProduct = np.dot(u, v)

Matrix math
from numpy.linalg import inv

A = np.matrix([[1,2,3,4],[5,6,7,8]])

# Multiply A by A transpose
AAt = np.dot(A, A.T)

# Invert AAt with np.linalg.inv()
AAtInv = np.linalg.inv(AAt)

# Show inverse times matrix equals identity
print ((AAtInv * AAt)

'''
PySpark provides a DenseVector class within the module pyspark.mllib.linalg.  
DenseVector is used to store arrays of values for use in PySpark.
DenseVector actually stores values in a NumPy array and delegates calculations to that object. 
You can create a new DenseVector using DenseVector() and passing in an NumPy array or a Python list
Dense vector stores values as float and uses .dot() to do dot product
'''
 
from pyspark.mllib.linalg import DenseVector
numpyVector = np.array([-3, -4, 5])
myDenseVector = DenseVector([3.0, 4.0, 5.0])
denseDotProduct = myDenseVector.dot(numpyVector)
 
