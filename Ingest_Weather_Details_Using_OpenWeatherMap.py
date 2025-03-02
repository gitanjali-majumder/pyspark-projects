# Databricks notebook source
# MAGIC %md
# MAGIC ### Ingest & Transform the Weather Data for different cities using OpenWeatherMap API

# COMMAND ----------

# importing required modules
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,to_timestamp,from_unixtime,StructType,explode_outer,ArrayType,to_date,max,date_format,expr,round,from_utc_timestamp,avg
import json
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

# COMMAND ----------

# Code for ingesting the data to dataframe
API_Key = 'd53513c201845dc6ea63950ce69f336d'
cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Surat", "Jaipur"
]

weather_date = []
for city in cities :
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_Key}'
    print("URL", url)
    response = requests.get(url)
    if response.status_code == 200:
        city_data = response.json()
        weather_date.append(city_data)
        todays_date = datetime.now()
        print(f"Weather Data for {todays_date}\n")

df = spark.read.json(spark.sparkContext.parallelize([json.dumps(entry) for entry in weather_date]))
display(df)


# COMMAND ----------

#Ingest Raw data to Delta Lake
df.write.format("delta").mode("append").saveAsTable("Daily_Weather_Forecast_Analysis")

# COMMAND ----------

#Store the Data from delta table to dataframe
df_raw = spark.sql(f"select * from daily_weather_forecast_analysis")
display(df_raw)

# COMMAND ----------

#Convert Epoch time to human readable time and removing any redundant data for current date
current_date = datetime.date(datetime.now(pytz.timezone("Asia/Kolkata")))
print("Today's Date:",current_date)
df_date_modfd = df_raw.withColumn("Date", from_unixtime(col("dt")).cast("timestamp"))\
                    .withColumn("Date",date_format(from_utc_timestamp("Date","Asia/Kolkata"),"yyyy-MM-dd HH:mm:dd"))\
                    .drop('dt').filter(to_date("Date")==current_date).dropDuplicates()
display(df_date_modfd)

# COMMAND ----------

# Find array columns and explode them
array_columns = [col.name for col in df_date_modfd.schema.fields if isinstance(col.dataType, ArrayType)]
for array_col in array_columns:
    df_exploded = df_date_modfd.withColumn(f"{array_col}",explode_outer(col(array_col)))
display(df_exploded)

# COMMAND ----------

#Find all the struct field and flatten them
struct_columns = [col.name for col in df_exploded.schema.fields if isinstance(col.dataType, StructType)]
struct_fields = [col(f"{struct_col}.{nested_cols}").alias(f"{struct_col}_{nested_cols}")
                 for struct_col in struct_columns
                 for nested_cols in df_exploded.select(struct_columns).schema[struct_col].dataType.names]
normal_column = [col(c) for c in df_exploded.columns if c not in struct_columns]
print("Struct Columns:", struct_columns,"\nNormal Columns:",normal_column,"\nStruct Fields",struct_fields)

df_flatten = df_exploded.select(*normal_column,*struct_fields)
display(df_flatten)

# COMMAND ----------

#Top 5 cloudy and humid cities in current date
df_renamed = df_flatten.select("Date","name","clouds_all","main_humidity","weather_description","main_temp_max","main_temp_min","main_feels_like")\
    .withColumnRenamed("name","City")\
        .withColumnRenamed("weather_description","Weather_Condition")\
            .withColumnRenamed("main_humidity" , "Humidity")\
                .withColumnRenamed("main_temp_max", "Max_Temp_Recorded")\
                    .withColumnRenamed("main_temp_min", "Min_Temp_Recorded")\
                        .withColumnRenamed("main_feels_like", "Feels_Like")\

df_cloudy = df_renamed.groupBy("Date","City","Weather_Condition","Humidity").agg(max("clouds_all").alias("Cloudiness_Percentage")).orderBy(max("clouds_all").desc()).limit(5)

display(df_cloudy)

# COMMAND ----------

# Sun-Rise & Sun-Set Timings in IST
df_sun_rise_set = df_flatten.select("Date","name","sys_sunrise","sys_sunset").withColumn("Date",to_date("Date"))\
                            .withColumnRenamed("name","City")\
                            .withColumn("Sunrise",from_unixtime("sys_sunrise").cast("timestamp"))\
                            .withColumn("Sunrise_IST",date_format(from_utc_timestamp("Sunrise","Asia/Kolkata"),"yyyy-MM-dd HH:mm:dd"))\
                            .withColumn("Sunset",from_unixtime("sys_sunset").cast("timestamp"))\
                            .withColumn("Sunset_IST",date_format(from_utc_timestamp("Sunset","Asia/Kolkata"),"yyyy-MM-dd HH:mm:dd"))\
                            .drop("sys_sunrise","sys_sunset","Sunrise","Sunset")\
                            .dropDuplicates()
display(df_sun_rise_set)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize using Matplotlib

# COMMAND ----------

# Top 3 Cities Experiencing more temp and high average temp throughout
df_temp_drop = df_renamed.groupBy("Date","City","Max_Temp_Recorded","Min_Temp_Recorded","Feels_Like").agg(avg((expr("Min_Temp_Recorded+Max_Temp_Recorded"))/2).alias("Average_Temp")).orderBy(avg((expr("Min_Temp_Recorded+Max_Temp_Recorded"))/2).desc())
df_temp_drop = df_temp_drop.withColumn("Max_Temp_in_Celsius",round(col("Max_Temp_Recorded")-273.15,2))\
                        .withColumn("Min_Temp_in_Celsius",round(col("Min_Temp_Recorded")-273.15,2))\
                        .withColumn("Feels_Like_in_Celsius",round(col("Feels_Like")-273.15,2))\
                        .withColumn("Average_Temp_in_Celsius",round(col("Average_Temp")-273.15,2))\
                        .drop("Max_Temp_Recorded","Min_Temp_Recorded","Feels_Like","Average_Temp") 
df_temp_drop = df_temp_drop.groupBy("City").agg(round(avg("Average_Temp_in_Celsius"),2).alias("Average_Temperature")).orderBy(avg("Average_Temp_in_Celsius").desc()).limit(3)              
display(df_temp_drop)

# COMMAND ----------

#Create a Bar Chart for the Temperature data
df_pandas= df_temp_drop.toPandas()

plt.figure(figsize=(5,6))
plt.bar(df_pandas['City'],df_pandas['Average_Temperature'],color ='orange')
plt.title("Top 3 cities experiencing high Average Temp in a Day")
plt.xlabel('City')
plt.ylabel('Average Temp in Celsius')
plt.xticks(rotation=0)
plt.show()

# COMMAND ----------

dbutils.notebook.exit("Success")
