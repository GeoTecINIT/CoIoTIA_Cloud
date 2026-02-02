import math
import folium
from folium.plugins import HeatMap

import pandas as pd
import numpy as np
from enum import Enum
import geopandas as gpd
import h3

from shapely import Point, unary_union
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity, NearestNeighbors
from shapely.geometry import box, Polygon, MultiPolygon
import branca.colormap as cm

# Resoluciones del 0 al 15 (0 = más grande, 15 = más pequeño)
# Definir un ratio a base de experimentación
# Limitar el uso a una area concreta para evitar demasiadas celdas

class CleanningSilhouetteProcess(Enum):
    CLIPPING = "clipping"
    POSITIVE_RESCALING = "positive_rescaling"
    RAW = "raw"

FILE_NAME_INPUT = "./KDE/datasets/clustering_with_elevation.csv"  # Ruta al archivo CSV con datos de sensores
FILE_CLIP_POLYGON = "./KDE/mapas/castellon_peninsula.geojson"  # Ruta al archivo GeoJSON con el polígono de recorte
FILE_NAME_OUPUT = "./KDE/mapas/clusters_h3_AVAMET_auto"  # Nombre base para archivos de salida
RES = 10   # Resolución H3
SILHOUETTE_THRESHOLD = 0.1  
LON_COL = "Longitude"
LAT_COL = "Latitude"
CLUSTER_COL = "Cluster"
SILHOUETTE_COL = "Silhouette Score"
SENSOR_ID_COL = "Sensor_ID"
BANDWIDTH = None # En metros, ajustar según escala de tus coordenadas
USE_WEIGHTS = True  # Usar pesos en el KDE
CLEANING_PROCESS = CleanningSilhouetteProcess.CLIPPING  # Proceso de limpieza de scores  

NO_CLUSTER_COLOR = "#f500d9"  # gris
NO_CLUSTER_OPACITY = 0.3

# --- Funciones auxiliares ---

def change_crs_to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lon_mean = gdf.geometry.x.mean()
    lat_mean = gdf.geometry.y.mean()

    utm_zone = math.floor((lon_mean + 180) / 6) + 1
    if lat_mean >= 0:
        epsg_code = 32600 + utm_zone  # North
    else:
        epsg_code = 32700 + utm_zone  # South

    gdf_utm = gdf.to_crs(epsg=epsg_code)
    gdf_utm["x"] = gdf_utm.geometry.x
    gdf_utm["y"] = gdf_utm.geometry.y

    return gdf_utm

def clean_silhouette(silhouette_scores: np.ndarray, process: CleanningSilhouetteProcess):

    if (silhouette_scores < 0).any() and process == CleanningSilhouetteProcess.RAW:
        raise ValueError(
            "Silhouette scores contain negative values, cannot use RAW process."
        )

    if process == CleanningSilhouetteProcess.CLIPPING:
        weights = np.clip(silhouette_scores, 0, None)
    elif process == CleanningSilhouetteProcess.POSITIVE_RESCALING:
        min_w = silhouette_scores.min()
        weights = silhouette_scores - min_w + 0.01
    elif process == CleanningSilhouetteProcess.RAW:
        weights = silhouette_scores
    return weights

def median_sensor_distance(coords: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=min(2, len(coords))).fit(coords)
    dists, _ = nn.kneighbors(coords)
    typical = np.median(dists[:, -1]) if len(coords) > 1 else 0.0
    return typical

def choose_bandwidth_for_cluster(
    coords: np.ndarray,
    weights: np.ndarray = None,
    min_points_cv: int = 10,
    n_grid: int = 15,
) -> float:

    # Median of nearest neighbor distances as typical scale
    typical = median_sensor_distance(coords)

    if len(coords) < min_points_cv or typical == 0:
        return typical

    # Create a bandwidth grid around the typical scale
    lo = 0.5 * typical
    hi = 3.0 * typical
    bw_grid = np.linspace(lo, hi, n_grid)

    kde = KernelDensity(kernel="gaussian")
    grid = GridSearchCV(
        kde,
        {"bandwidth": bw_grid},
        cv=min(5, len(coords)),  # CV = menor de 5 o n
        n_jobs=-1,
    )

    if weights is not None:
        grid.fit(coords, sample_weight=weights)
    else:
        grid.fit(coords)

    return grid.best_params_["bandwidth"]

def style_function(feature):
    cluster = feature["properties"]["cluster"]

    # Caso especial: no_cluster
    if cluster == "no_cluster":
        return {
            "fillColor": NO_CLUSTER_COLOR,
            "color": "#555555",
            "weight": 1,
            "fillOpacity": NO_CLUSTER_OPACITY,
        }

    # Resto de clusters normales
    idx = cluster_to_idx[cluster]
    return {
        "fillColor": colormap(idx),
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.6,
    }

# --- Funciones principales ---

def read_clustering_file(filepath: str, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:

    # Get dataframe from CSV
    sensors = pd.read_csv(filepath)

    # Get GeoDataFrame with geometry from lon/lat
    gdf = gpd.GeoDataFrame(
        sensors,
        geometry=gpd.points_from_xy(sensors[lon_col], sensors[lat_col]),
        crs="EPSG:4326",
    )

    return gdf

def get_h3_cells(gdf: gpd.GeoDataFrame, clip_poly: Polygon = None) -> gpd.GeoDataFrame:

    # If not provided, use the bounding box of the gdf
    if clip_poly is None:
        minx, miny, maxx, maxy = gdf.total_bounds
        clip_poly = box(minx, miny, maxx, maxy)

    # If clip_poly is Polygon or MultiPolygon, convert to H3 LatLngPoly or LatLngMultiPoly
    if isinstance(clip_poly, Polygon):
        h3_shape = h3.LatLngPoly(
            outer=[(lat, lon) for lon, lat in clip_poly.exterior.coords],
            *[[(lat, lon) for lon, lat in ring.coords] for ring in clip_poly.interiors],
        )
    elif isinstance(clip_poly, MultiPolygon):
        polys = []
        for poly in clip_poly.geoms:
            outer = [(lat, lon) for lon, lat in poly.exterior.coords]
            interiors = [
                [(lat, lon) for lon, lat in ring.coords] for ring in poly.interiors
            ]
            polys.append(h3.LatLngPoly(outer=outer, *interiors))
        h3_shape = h3.LatLngMultiPoly(*polys)

    else:
        raise ValueError("clip_poly debe ser Polygon o MultiPolygon")

    cells_ids = h3.polygon_to_cells(h3_shape, RES)
    cells_centroids = [h3.cell_to_latlng(cell) for cell in cells_ids]

    cent_gdf = gpd.GeoDataFrame(
        {"cell": cells_ids},
        geometry=[Point(lon, lat) for lat, lon in cells_centroids],
        crs="EPSG:4326",
    )

    return cent_gdf

def get_clusters_KDE(
    sensores: gpd.GeoDataFrame,
    bandwidth: float = BANDWIDTH,
    cluster_col: str = "cluster",
    silhouette_col: str = "silhouette",
    noise_label: int = -1,
    use_weights: bool = True,
    cleanning_process: CleanningSilhouetteProcess = CleanningSilhouetteProcess.CLIPPING,
) -> dict:

    # Obtenemos las etiquetas de los clusters, menos de los ruidos
    clusters = sensores[cluster_col].unique()
    clusters = clusters[clusters != noise_label]

    kde_models = {}

    for cluster in clusters:

        cluster_sensors = sensores[sensores[cluster_col] == cluster]
        coords = cluster_sensors[["x", "y"]].to_numpy()

        if bandwidth is None:
            bandwidth = choose_bandwidth_for_cluster(coords)

        print(f"Cluster {cluster}, bandwidth elegido: {bandwidth:.1f} m")
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        if use_weights:
            silhouette_scores = cluster_sensors[silhouette_col].to_numpy()
            weights = clean_silhouette(silhouette_scores, cleanning_process)
            kde.fit(coords, sample_weight=weights)
        else:
            kde.fit(coords)

        kde_models[cluster] = kde

    return kde_models

def get_influence_areas(cent_gdf_m: gpd.GeoDataFrame, kde_models: dict, noise_label: str = "no_cluster", threshold: float = 0.0) -> pd.DataFrame:

    influence = pd.DataFrame({"cell": cent_gdf_m["cell"].values})

    centrois_list = [(pt.x, pt.y) for pt in cent_gdf_m.geometry]
    for c, kde in kde_models.items():
        log_dens = kde.score_samples(centrois_list)
        dens = np.exp(log_dens)
        influence[f"kde_{c}"] = dens

    kde_cols = [col for col in influence.columns if col.startswith("kde_")]
    influence["max_kde"] = influence[kde_cols].max(axis=1)
    influence["dominant_cluster"] = np.where(
        influence["max_kde"] <= threshold,
        noise_label, 
        influence[kde_cols].idxmax(axis=1).str.replace("kde_", "")
    )
    influence.drop(columns=["max_kde"], inplace=True)

    return influence

def get_clusters_polygons(influence: pd.DataFrame) -> gpd.GeoDataFrame:
    
    # For every cluster get a list of cells it dominates
    cells_by_cluster = (
        influence[["cell", "dominant_cluster"]]
        .groupby("dominant_cluster")["cell"]
        .apply(list)
        .to_dict()
    )

    # Tranform individual cells to lat/lon polygons
    rows = []
    for cluster, cells in cells_by_cluster.items():
        for h3_id in cells:
            poly = Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h3_id)])
            rows.append({"cluster": cluster, "h3_id": h3_id, "geometry": poly})
    gdf_clusters = gpd.GeoDataFrame(rows)
    gdf_clusters.set_geometry("geometry", inplace=True)
    gdf_clusters.set_crs(epsg=4326, inplace=True)

    # Dissolve polygons of each cluster
    gdf_unioned = gdf_clusters.dissolve(by="cluster", as_index=False)
    
    return gdf_unioned

# --- Ejecución principal ---

if __name__ == "__main__":

    # Read sensors data and the clip polygon
    filepath = FILE_NAME_INPUT  # = sys.argv[1]
    sensors_gdf = read_clustering_file(filepath, lon_col=LON_COL, lat_col=LAT_COL)
    sensors_gdf = sensors_gdf[sensors_gdf[SILHOUETTE_COL] >= SILHOUETTE_THRESHOLD] # OPCIONAL: Filtrar por silhouette mínimo
    sensors_gdf_m = change_crs_to_utm(sensors_gdf)
    cp = gpd.read_file(FILE_CLIP_POLYGON)

    # Generate centroids (with ids) of H3 cells covering the area
    cent_gdf = get_h3_cells(sensors_gdf, clip_poly=unary_union(cp.geometry))
    cent_gdf_m = change_crs_to_utm(cent_gdf)
    print("Celdas ya creadas")

    # Get the KDE models per cluster
    kde_models = get_clusters_KDE(sensors_gdf_m, bandwidth=BANDWIDTH, cluster_col=CLUSTER_COL, silhouette_col=SILHOUETTE_COL, use_weights=USE_WEIGHTS, cleanning_process=CLEANING_PROCESS)
    print("KDE calculado")

    # Get the most influential cluster per cell
    influence = get_influence_areas(cent_gdf_m, kde_models)

    # Eliminate cells with no dominant cluster if needed
    # influence = influence[influence["dominant_cluster"] != "no_cluster"]

    # Generate polygons per cluster
    gdf_unioned = get_clusters_polygons(influence)
    print("Poligonos por cluster disueltos")

    # DESECHABLE: Visualización con Folium
    clusters = sorted(gdf_unioned["cluster"].unique())

    colormap = cm.LinearColormap(
        colors=[
            "#440154", "#3b528b", "#21918c",
            "#5ec962", "#fde725"
        ],
        vmin=min(range(len(clusters))),
        vmax=max(range(len(clusters))),
    )

    # Mapear cluster → índice numérico
    cluster_to_idx = {cluster: i for i, cluster in enumerate(clusters)}

    m = folium.Map(
        location=[sensors_gdf[LAT_COL].mean(), sensors_gdf[LON_COL].mean()], zoom_start=12
    )  

    folium.GeoJson(
        gdf_unioned,
        name="Clusters Dominantes",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["cluster"],
            aliases=["Cluster:"],
        ),
    ).add_to(m)

    colormap.caption = "Clusters"
    colormap.add_to(m)

    for _, row in sensors_gdf.iterrows():
        folium.Marker(
            location=[row[LAT_COL], row[LON_COL]],
            tooltip=f"{row.get(SENSOR_ID_COL,'Sensor')}<br>Cluster: {row[CLUSTER_COL]}<br>Silhouette: {row[SILHOUETTE_COL]:.2f}",
            icon=folium.Icon(icon="traffic-light", color="blue", prefix="fa"),
    ).add_to(m)

    if USE_WEIGHTS:
        m.save(FILE_NAME_OUPUT + "_with_weights.html")
    else:
        m.save(FILE_NAME_OUPUT + "_no_weights.html")
