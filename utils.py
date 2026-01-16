import pandas as pd
import geopandas as gpd

def load_regions(regions):
    gdf = gpd.GeoDataFrame.from_features(regions)
    return gdf

def load_sensors(sensors):
    df = pd.DataFrame(sensors)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']))
    return gdf

def determine_sensor_region(regions, sensors):
    regions_gdf = load_regions(regions)
    sensors_gdf = load_sensors(sensors)
    sensors_regions = gpd.sjoin(sensors_gdf, regions_gdf, how="left", predicate='within')
    sensors_regions = sensors_regions.rename(columns={"name": "region"})
    return sensors_regions[["id", "lat", "lon", "virtual", "mobile", "region"]].to_json(orient="records")


def group_sensors_by_region(sensors):
    df = pd.DataFrame(sensors)
    groups = df.groupby(by="region").apply(lambda x: x.to_json(orient="records"))
    return groups