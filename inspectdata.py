import numpy as np
import plotly.express as px
from shapely import wkt

asset_indexes = np.load("asset_indexes.npy") 
distance_matrix = np.load("distance_matrix.npy") 
photo_indexes = np.load("photo_indexes.npy") 
points_lat_long = np.load("points_lat_long.npy") 
predecessors = np.load("predecessors.npy") 
waypoint_indexes = np.load("waypoint_indexes.npy") 
polygon_wkt = open("polygon_lon_lat.wkt").read()

print("asset_indexes shape:", asset_indexes.shape)
print("distance_matrix shape:", distance_matrix.shape)
print("photo_indexes shape:", photo_indexes.shape)
print("points shape:", points_lat_long.shape)
print("predecessors shape:", predecessors.shape)

N = points_lat_long.shape[0]
invalid_assets = [i for i in asset_indexes if i >= N or i < 0]
invalid_photos = [i for i in photo_indexes if i >= N or i < 0]

print(f"Invalid asset indexes: {invalid_assets}")
print(f"Invalid photo indexes: {invalid_photos}")


fig = px.scatter_mapbox(
    lat=points_lat_long[:,1],
    lon=points_lat_long[:,0],
    zoom=12,
    title="Raw Waypoints"
)

if polygon_wkt:
    poly = wkt.loads(polygon_wkt)
    x, y = zip(*poly.exterior.coords)
    fig.add_scattermapbox(
        lon=x, lat=y,
        mode="lines",
        name="Allowed Polygon",
        line=dict(color="black", width=2)
    )

fig.update_layout(
    mapbox_style="open-street-map",
    map_center = {
    "lon": float(points_lat_long[:,0].mean()), 
    "lat": float(points_lat_long[:,1].mean())   
},
    height=700,
    margin={"r":0,"t":0,"l":0,"b":0}
)

fig.show()

# print(asset_indexes) 
# print(distance_matrix) 
# print(photo_indexes) 
# print(points_lat_long) 
# print(predecessors) 
# print(waypoint_indexes)
# print(polygon_wkt)