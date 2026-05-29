import pandas as pd
import geopandas as gpd

def load_regions(regions):
    gdf = gpd.GeoDataFrame.from_features(regions, crs="EPSG:4326")
    return gdf

def load_sensors(sensors):
    df = pd.DataFrame(sensors)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
    return gdf

def determine_sensor_region(regions, sensors):
    regions_gdf = load_regions(regions)
    sensors_gdf = load_sensors(sensors)
    sensors_regions = gpd.sjoin(sensors_gdf, regions_gdf, how="left", predicate='within')
    sensors_regions = sensors_regions.rename(columns={"name": "region"})
    base_columns = ["id", "lat", "lon", "virtual", "mobile", "region"]
    for col in ["federated", "active"]:
        if col in sensors_regions.columns:
            base_columns.append(col)
    return sensors_regions[base_columns].to_json(orient="records")


def group_sensors_by_region(sensors):
    df = pd.DataFrame(sensors)
    if "region" not in df.columns:
        df["region"] = "default"
    groups = df.groupby(by="region").apply(lambda x: x.to_json(orient="records"))
    return groups