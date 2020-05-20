import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import elevation_grid as eg
from pyspark.sql import SparkSession, functions, types
from pyspark.ml import PipelineModel
from datetime import datetime
spark = SparkSession.builder.appName('Weather plot').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

def main(input,model):
    tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
    ])

    data = spark.read.csv(input, schema=tmax_schema)
    data.createOrReplaceTempView("data")
    extract_year = spark.sql("SELECT *,YEAR(date) AS year FROM data")
    extract_year.createOrReplaceTempView("extract_year")
    years_range1_data = spark.sql("SELECT * FROM extract_year WHERE year BETWEEN 1950 AND 1980")
    years_range1_data.createOrReplaceTempView("years_range1_data")
    years_range2_data = spark.sql("SELECT * FROM extract_year WHERE year BETWEEN 1981 AND 2010")
    years_range2_data.createOrReplaceTempView("years_range2_data")
    avg_range1_data = spark.sql("SELECT station,latitude,longitude,AVG(tmax) AS avg_tmax FROM years_range1_data GROUP BY station,latitude,longitude")
    avg_range2_data = spark.sql("SELECT station,latitude,longitude,AVG(tmax) AS avg_tmax FROM years_range2_data GROUP BY station,latitude,longitude")
    mapa(avg_range1_data,"Fig1","1950-1980")
    mapa(avg_range2_data,"Fig2","1981-2010")
# ---------------------
    lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
    elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]
    latitude = lats.reshape(1,-1)[0]
    longitude = lons.reshape(1,-1)[0]
    elevation = np.reshape(elevs, (720,360)).reshape(1,-1)[0]
    elevation_data = pd.DataFrame({'latitude':latitude,'longitude':longitude,'elevation':elevation})
    elevation_data = spark.createDataFrame(elevation_data).withColumn("date",functions.lit(datetime.strptime('2019-08-09','%Y-%m-%d').date())).withColumn("tmax",functions.lit(0))

    model_weather = PipelineModel.load(model)
    predictions = model_weather.transform(elevation_data)
    mapb1(predictions)
# ---------------------
    test_data = spark.read.csv("tmax-test",schema = tmax_schema)
    test_data = model_weather.transform(test_data)
    test_error_data = test_data.withColumn("prediction_error",functions.abs(test_data["tmax"]-test_data["prediction"]))
    mapb2(test_error_data)
# ---------------------
#
def mapa(weather_df,figure_name,year_range):
    weather_data = weather_df.toPandas()
    tmax = weather_data["avg_tmax"].values
    latitude = weather_data["latitude"].values
    longitude = weather_data["longitude"].values

    figure = plt.figure(figure_name)

    basemap = Basemap(projection = "cyl",urcrnrlat=80,urcrnrlon=180,llcrnrlat=-80,llcrnrlon=-180,resolution = 'c')
    basemap.shadedrelief(scale=0.2)
    basemap.drawcoastlines(color="black",linewidth=0.1)
    basemap.drawmapboundary(color="black",linewidth=0.5)

    x,y = basemap(longitude,latitude)
    plt.scatter(x,y,cmap="plasma",marker="o",c=tmax,s=1)
    plt.colorbar(label="average temperature (in celcius)")
    plt.title("Global Temperature Change "+year_range)
    plt.savefig(figure_name+".jpg",dpi=400)


def mapb1(temp_df):
    temp_data = temp_df.toPandas()
    temp = temp_data["prediction"].values
    latitude = temp_data["latitude"].values
    longitude = temp_data["longitude"].values

    fig = plt.figure("mapb1")
    basemap = Basemap(projection='robin',lon_0 = 0,resolution = 'c')
    basemap.drawcoastlines(color = 'black',linewidth = 0.5)
    basemap.drawmapboundary(color = 'black',linewidth = 0.5)
    x,y = basemap(longitude,latitude)
    plt.scatter(x,y,c=temp,s=2,marker='o',cmap='Paired_r')
    plt.colorbar(label = "temperature in celcius")
    plt.title("Temperature predictions on 09/08/2019")
    plt.savefig("prediction.jpg",dpi = 400)

def mapb2(error_df):
    error_data = error_df.toPandas()
    error = error_data["prediction_error"].values
    latitude = error_data["latitude"].values
    longitude = error_data["longitude"].values

    figure = plt.figure("mapb2")

    basemap = Basemap(projection = "cyl",urcrnrlat=80,urcrnrlon=180,llcrnrlat=-80,llcrnrlon=-180,resolution = 'c')
    basemap.shadedrelief(scale=0.2)
    basemap.drawcoastlines(color="black",linewidth=0.1)
    basemap.drawmapboundary(color="black",linewidth=0.5)

    x,y = basemap(longitude,latitude)
    plt.scatter(x,y,cmap="Paired_r",marker="o",c=error,s=1)
    plt.colorbar(label="error value")
    plt.title("Test Error on prediction of tmax values")
    plt.savefig("error.jpg",dpi=400)


if __name__ == '__main__':
    input = sys.argv[1]
    model = sys.argv[2]
    main(input,model)
