'''
log data comes from many sources, such as web, file, and compute servers, application logs, user-generated content, and can be used for monitoring servers,
 improving business and customer intelligence, building recommendation systems, fraud detection, and much more.
Server log analysis is an ideal use case for Spark. It's a very large, common data source and contains a rich set of information. 
Spark allows you to store your logs in files on disk cheaply, while still providing a quick and simple way to perform data analysis on them
'''


#Exploratory Data Analysis
#Importing Libraries
import re
import datetime
from databricks_test_helper import Test
import sys
import os

#Loading the input file using Sql context . This reads in the schema as well

log_file_path = 'dbfs:/' + os.path.join('databricks-datasets', 'cs100', 'lab2', 'data-001', 'apache.access.log.PROJECT')
base_df = sqlContext.read.text(log_file_path)

#Looking at some of the data
base_df.show(truncate=False)
base_df.filter(base_df['value'].isNull()).count() # Verify that there are no null rows in the original data set.

'''
Common log format for webservers

remotehost rfc931 authuser [date] "request" status bytes
field	meaning
remotehost	Remote hostname (or IP number if DNS hostname is not available).
rfc931	The remote logname of the user. We don't really care about this field.
authuser	The username of the remote user, as authenticated by the HTTP server.
[date]	The date and time of the request.
"request"	The request, exactly as it came from the browser or client.
status	The HTTP status code the server sent back to the client.
bytes	The number of bytes (Content-Length) transferred to the client.
'''

# Using regular expression to parse the data

from pyspark.sql.functions import split, regexp_extract
split_df = base_df.select(regexp_extract('value', r'^([^\s]+\s)', 1).alias('host'),
                          regexp_extract('value', r'^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]', 1).alias('timestamp'),
                          regexp_extract('value', r'^.*"\w+\s+([^\s]+)\s+HTTP.*"', 1).alias('path'),
                          regexp_extract('value', r'^.*"\s+([^\s]+)', 1).cast('integer').alias('status'),
                          regexp_extract('value', r'^.*\s+(\d+)$', 1).cast('integer').alias('content_size'))
split_df.show(truncate=False)


# Check the number of rows null column values
from pyspark.sql.functions import col, sum

def count_null(col_name):
  return sum(col(col_name).isNull().cast('integer')).alias(col_name)

# Build up a list of column expressions, one per column.

exprs = []
for col_name in split_df.columns:
  exprs.append(count_null(col_name))

# Run the aggregation. The *exprs converts the list of expressions into
# variable function arguments.
split_df.agg(*exprs).show()

# It appears that null values was sent by the server which we want to map to zero so that they are counted properly

# Replace all null content_size values with 0.
cleaned_df = split_df.na.fill({'content_size': 0})

# Ensure that there are no nulls left.
exprs = []
for col_name in cleaned_df.columns:
  exprs.append(count_null(col_name))

cleaned_df.agg(*exprs).show()

 # Using User-Defined Function (UDF) to parse the time stamp
 
 month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(s):
    """ Convert Common Log time format into a Python datetime object
    Args:
        s (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring time zone here. In a production application, you'd want to handle that.
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(s[7:11]),
      month_map[s[3:6]],
      int(s[0:2]),
      int(s[12:14]),
      int(s[15:17]),
      int(s[18:20])
    )

u_parse_time = udf(parse_clf_time)

logs_df = cleaned_df.select('*', u_parse_time(cleaned_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp')
total_log_entries = logs_df.count()

logs_df.printSchema()
display(logs_df)

# Caching DF as it will be re-used
logs_df.cache()

#Analysis on the Web Server Log File

# Calculate statistics based on the content size.
content_size_summary_df = logs_df.describe(['content_size'])
content_size_summary_df.show()

# Getting summary statistics using SQL function
from pyspark.sql import functions as sqlFunctions
content_size_stats =  (logs_df
                       .agg(sqlFunctions.min(logs_df['content_size']),
                            sqlFunctions.avg(logs_df['content_size']),
                            sqlFunctions.max(logs_df['content_size']))
                       .first())

print 'Using SQL functions:'
print 'Content Size Avg: {1:,.2f}; Min: {0:.2f}; Max: {2:,.0f}'.format(*content_size_stats)

#HTTP Status Analysis ie Status code vs Count

status_to_count_df =(logs_df
                     .groupBy('status')
                     .count()
                     .sort('status')
                     .cache())

status_to_count_length = status_to_count_df.count()
print 'Found %d response codes' % status_to_count_length
status_to_count_df.show()

#visualize the results from the last example
display(status_to_count_df)

log_status_to_count_df = status_to_count_df.withColumn('log(count)', sqlFunctions.log(status_to_count_df['count']))
display(log_status_to_count_df) #Using Log so that the dristribution is better

# Looking at hosts that have accessed the server frequently (e.g., more than ten times)

# Any hosts that has accessed the server more than 10 times.
host_sum_df =(logs_df
              .groupBy('host')
              .count())

host_more_than_10_df = (host_sum_df
                        .filter(host_sum_df['count'] > 10)
                        .select(host_sum_df['host']))

print 'Any 20 hosts that have accessed more then 10 times:\n'
host_more_than_10_df.show(truncate=False)


#Number of Unique Hosts

unique_host_count = (logs_df.select('host').distinct().count())
print 'Unique hosts: {0}'.format(unique_host_count)

#explore the error 404 status records
 
not_found_df = logs_df.filter(logs_df['status'] == 404)
print('Found {0} 404 URLs').format(not_found_df.count())

#Printing list up to 40 distinct paths that generate 404 errors. 

not_found_paths_df = not_found_df.groupBy('path').count().sort(desc('count'))
unique_not_found_paths_df = not_found_paths_df

print '404 URLS:\n'
unique_not_found_paths_df.show(n=40, truncate=False)




