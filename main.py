# Main Library Imports
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from shapely import wkt
import os

# Custom Imports
from tsp_solver import ortools_single_tsp, route_distance

# Optional: quick map (will be skipped if geopandas/contextily not installed)
def quick_map(coords, asset_idx, photo_idx, polygon_wkt, out_png="out/step1_overview.png"):
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import contextily as cx
        from shapely.geometry import Point
    except Exception as e:
        print(f"[map] Skipping map (missing deps): {e}")
        return

    poly = wkt.loads(polygon_wkt)
    poly_gdf = gpd.GeoDataFrame({"name": ["flight_zone"], "geometry": [poly]}, crs="EPSG:4326").to_crs(epsg=3857)

    all_points = gpd.GeoDataFrame(
        {"idx": np.arange(len(coords))},
        geometry=[Point(lon, lat) for lon, lat in coords],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    assets_gdf = all_points.iloc[asset_idx] if len(asset_idx) else all_points.iloc[[]]
    photos_gdf = all_points.iloc[photo_idx] if len(photo_idx) else all_points.iloc[[]]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    poly_gdf.boundary.plot(ax=ax, linewidth=1.2, alpha=0.9, label="Flight Zone")
    try:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    except Exception:
        pass
    if len(assets_gdf): assets_gdf.plot(ax=ax, markersize=6, alpha=0.7, label="Assets (ref)", color="gray")
    if len(photos_gdf): photos_gdf.plot(ax=ax, markersize=8, alpha=0.9, label="Photo targets", color="tab:blue")
    ax.set_title("Step 1: Flight Zone & Waypoints")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"[map] Saved quick-look map → {out_png}")

# Load all the data
data = {}
data_dir = "data"

data["assets"] = np.load(f"{data_dir}/asset_indexes.npy")
data["distances"] = np.load(f"{data_dir}/distance_matrix.npy")
data["photo"] = np.load(f"{data_dir}/photo_indexes.npy")
data["points"] = np.load(f"{data_dir}/points_lat_long.npy")
data["predecessors"] = np.load(f"{data_dir}/predecessors.npy")
data["waypoints"] = np.load(f"{data_dir}/waypoint_indexes.npy")

with open(f"{data_dir}/polygon_lon_lat.wkt") as f:
  polygon_wkt = f.read().strip()

D = data["distances"]
P = data["predecessors"]
coords = data["points"]

# Sanity Check
N = D.shape[0]
print("=== SHAPES ===")
print("distance_matrix:", D.shape)
print("predecessors:   ", P.shape)
print("points (lon,lat):", coords.shape)
if data["waypoints"] is not None:
    print("waypoint_indexes:", data["waypoints"].shape)

assert D.shape == (N, N), "Distance matrix must be square"
assert P.shape[0] == N, "predecessors first dimension must match distance_matrix size"
assert coords.shape[1] == 2, "points_lat_long must have shape (K, 2)"

# Slice the arrays defensively
def slice_to_indices(arr, N):
   start = int(arr[0])
   end = int(arr[-1])

   if end <= start:
      return np.array([], dtype=int), (start, end, 0)
   raw = np.arange(start, end, dtype=int)
   oob_mask = (raw < 0) | (raw >= N)
   kept = raw[~oob_mask]
   return kept, (start, end, int(oob_mask.sum()))

asset_idx, asset_info = slice_to_indices(data["assets"], N)
photo_idx, photo_info = slice_to_indices(data["photo"], N)

asset_coords = coords[3544:4274]  # plot-only

print("\n=== INDEX RANGES ===")
print(f"Assets slice raw:  [{asset_info[0]}, {asset_info[1]})  -> kept {len(asset_idx)} (oob dropped: {asset_info[2]})")
print(f"Photos slice raw:  [{photo_info[0]}, {photo_info[1]})  -> kept {len(photo_idx)} (oob dropped: {photo_info[2]})")

# Directional Matrix Check

sym_ok = np.allclose(D, D.T, atol=1e-6, rtol=1e-6)
finite_ok = np.isfinite(D).all()
nonneg_ok = (D >= 0).all()

D_off = D.copy()
np.fill_diagonal(D_off, np.inf)
min_off = float(np.min(D_off))
max_any = float(np.max(D))
print(f"Min off-diag ft: {min_off:.3f}")
print(f"Max any ft:     {max_any:.3f}")

print("\n=== DISTANCE MATRIX CHECKS ===")
print("Symmetric:      ", sym_ok)
print("Finite values:  ", finite_ok)
print("Non-negative:   ", nonneg_ok)
print(f"Min off-diag ft: {min_off:.3f}")
print(f"Max any ft:      {max_any:.3f}")


# ---------- Small summary table ----------
summary = pd.DataFrame(
    {
        "metric": [
            "N (nodes)",
            "# asset indices (kept)",
            "# photo indices (kept)",
            "D symmetric?",
            "D finite?",
            "D non-negative?",
            "min off-diag (ft)",
            "max (ft)",
        ],
        "value": [
            N,
            len(asset_idx),
            len(photo_idx),
            sym_ok,
            finite_ok,
            nonneg_ok,
            f"{min_off:.3f}",
            f"{max_any:.3f}",
        ],
    }
)
print("\n=== SUMMARY ===")
print(summary.to_string(index=False))

# ---------- Optional quick-look map ----------
#quick_map(coords, asset_idx, photo_idx, polygon_wkt, out_png="out/step1_overview.png")

# TSP solver
D = data["distances"]
N = D.shape[0]

# Targets: all photo waypoints except depot (0)
photo_targets = np.arange(0, N, dtype=int)
photo_targets = photo_targets[photo_targets != 0]
depot = 0

# Build tour
tsp_route = ortools_single_tsp(D, depot=depot)

dist_ft = route_distance(tsp_route, D)
dist_mi = dist_ft / 5280.0

print(f"TSP route length: {len(tsp_route)} nodes")
print(f"Total tour distance: {dist_ft:.1f} ft ({dist_mi:.2f} mi)")
print(f"Route starts at {tsp_route[0]}, ends at {tsp_route[-1]}")
