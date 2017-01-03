''' 
The volume of unstructured text in existence is growing dramatically, and Spark is an excellent tool for analyzing this type of data.
Here we will write code that calculates the most common words in the Complete Works of William Shakespeare retrieved from Project Gutenberg

'''

#Sample exercises
wordsDF = sqlContext.createDataFrame([('cat',), ('elephant',), ('rat',), ('rat',), ('cat', )], ['word'])
wordsDF.show()
print type(wordsDF)
wordsDF.printSchema()

#Concantenate words with s

from pyspark.sql.functions import lit, concat

pluralDF = wordsDF.select(concat(wordsDF.word,lit("s")).alias("word"))
pluralDF.show()

#Length of each word

from pyspark.sql.functions import length
pluralLengthsDF = pluralDF.select(length(pluralDF.word))
pluralLengthsDF.show()

#count the number of times a particular word appears in the 'word' column
wordCountsDF = (wordsDF.groupBy(wordsDF.word)).count()
wordCountsDF.show()

#Calculate the number of unique words in wordsDF

uniqueWordsCount =  wordCountsDF.count()
print uniqueWordsCount

#Find the mean number of occurrences of words in wordCountsDF.

averageCount = (wordCountsDF.groupBy().mean()).collect()[0][0]

print averageCount



# Define a function for word counting

def wordCount(wordListDF):
    """Creates a DataFrame with word counts
    Args: DataFrame consisting of one string column called 'word'.
    Returns: DataFrame of (str, int):containing 'word' and 'count' columns.
    """
    return (wordListDF.groupBy(wordListDF.word)).count()



# Use regular expressions to remove punctuation , change to lower case and strip leading and trailing spaces

from pyspark.sql.functions import regexp_replace, trim, col, lower
def removePunctuation(column):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces.
      Args: Column containing a sentence
    Returns: Column named 'sentence' with clean-up operations applied.
    """
    colNoPunct = regexp_replace(column, "[^a-zA-Z0-9 ]", "")
    #colNoPunct = regexp_replace(column, string.punctuation, "") .
    #^ specifies not the mentioned characters and replacing with ""    
    trim_lower = trim(lower(colNoPunct)) #Trim trailing space and to lower case
    return(trim_lower) 
		
# Load Text File from web link	
fileName = "dbfs:/databricks-datasets/cs100/lab1/data-001/shakespeare.txt"

shakespeareDF = sqlContext.read.text(fileName).select(removePunctuation(col('value')))
shakespeareDF.show(15, truncate=False)

# Apply a transformation that will split each 'sentence' in the DataFrame by its spaces
# Then transform from a DataFrame that contains lists of words into a DataFrame with each word in its own row. 
from pyspark.sql.functions import split, explode
# shakeWordsDF = (shakespeareDF
#                 .select(shakespeareDF[0].alias('sentence'))
#                ) 
shakeWordsDF = (shakespeareDF
                .select(split(shakespeareDF[0], " ").alias('wordLst'))
               )
shakeWordsDF = (shakeWordsDF
                .select(explode(shakeWordsDF.wordLst).alias('word'))
                .where("word != ''")
               )

shakeWordsDF.show()
shakeWordsDFCount = shakeWordsDF.count()
print shakeWordsDFCount

#Count the Number of words
from pyspark.sql.functions import desc
topWordsAndCountsDF = wordCount(shakeWordsDF).orderBy(desc('count'))
topWordsAndCountsDF.show() #Top counts are stopwords


