# Main Library Imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from shapely import wkt
from shapely.geometry import Point, LineString
from pathlib import Path
import os, json

# Custom Imports
from data_metrics import display_data_metrics
from tsp_solver import ortools_single_tsp, route_distance
from predecessor_formatter import get_waypoints_to_nav_map, expand_route_csgraph, export_expanded_coords
from heuristics import greedy_split_by_battery
from visualizer import load_mission_polylines, percent_length_outside

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

# Display the data metrics: optional
#display_data_metrics(data)

# TSP solver
D = data["distances"]
P = data["predecessors"]
N = D.shape[0]
coords = data["points"]

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

# Step 3 (expand the predecessors.npy file)
N_wp = D.shape[0]
N_nav = P.shape[1]

wp2nav = get_waypoints_to_nav_map(N_wp, N_nav, waypoints_mapping=None)

expanded_nav = expand_route_csgraph(tsp_route, P, wp2nav)
_ = export_expanded_coords(expanded_nav, coords, out_csv_path="out/tsp_expanded_path.csv")

print(f"[Step3] Expanded nav-node path length: {len(expanded_nav)}")

# Step 4 (Heuristics to optimize path)
BATTERY_FT = 37725.0
SAFETY_BUFFER = 0.95
CAP_FT = BATTERY_FT * SAFETY_BUFFER

out_dir = "out"
os.makedirs(out_dir, exist_ok=True)

# Split the missions (We can change the heuristics here)
missions = greedy_split_by_battery(tsp_route, D, CAP_FT, depot=0)
print(f"[Step4] Missions created: {len(missions)} (cap {CAP_FT:.1f} ft incl. buffer {SAFETY_BUFFER*100:.0f}%)\n")

# Expand each mission using the expanded predecessors data
wp2nav = get_waypoints_to_nav_map(D.shape[0], P.shape[1], waypoints_mapping=None)

# Record the metrics of each mission
mission_metrics = []
for k, mission in enumerate(missions):
  # Expand the nav-node path
  expanded_nav = expand_route_csgraph(mission, P, wp2nav)

  # Exporting settings
  csv_path = f"{out_dir}/mission_{k:02d}.csv"
  df = export_expanded_coords(expanded_nav, coords, out_csv_path=csv_path)

  # Compute the abstract routing distance for the mission
  dist_ft = route_distance(mission, D)
  dist_mi = dist_ft/5280.0

  mission_metrics.append({
    "mission": k,
    "num_waypoints": max(0, len(mission) - 2),
    "distance_ft" : float(dist_ft),
    "distance_mi" : float(dist_mi),
    "csv": Path(csv_path).name
  })

print(f"\n[Step4] Expanded & exported {len(missions)} missions to {out_dir}/")

# Create a summary table of all the missions
summary_df = pd.DataFrame(mission_metrics)
summary_df.to_csv(f"{out_dir}/missions_summary.csv", index=False)

print("[Step4] Summary:")
print(summary_df.head(min(10, len(summary_df))))
print(f"[Step4] Saved → {out_dir}/missions_summary.csv")

# Optional: quick overall stats
total_ft = float(summary_df["distance_ft"].sum()) if len(summary_df) else 0.0
print(f"[Step4] Total routing distance across missions: {total_ft:.1f} ft "
      f"({total_ft/5280.0:.2f} mi)")

# Step 5 Process all the data into points for visualization
depot = 0
all_visited = set()
for m in missions:
  all_visited.update([x for x in m if x != depot])

photo_targets = np.arange(0, D.shape[0], dtype=int)
photo_targets = photo_targets[photo_targets != depot]
missed = sorted(set(photo_targets) - all_visited)

print("\n[Step5] Coverage Check")
print(f"Visited: {len(all_visited)} / {len(photo_targets)}")
print("Missed targets:", "NONE" if not missed else missed)

# Recheck for any battery constraint errors
cap_report = []
violations = []
for k, m in enumerate(missions):
  dist_ft = route_distance(m, D)
  valid = dist_ft <= CAP_FT + 1e-6
  cap_report.append({
    "mission": k,
    "distance_ft": dist_ft,
    "valid": valid,
    "slack_ft": CAP_FT - dist_ft
  })

  if not valid:
    violations.append((k, dist_ft))
if violations:
  print("[Step5] Battery violations found:", violations)
else:
  print("[Step5] All missions within cap!")

mission_ids = []
mission_geoms = []
for k in range(len(missions)):
    csv_file = f"{out_dir}/mission_{k:02d}.csv"
    if not os.path.exists(csv_file):
        print(f"[Step5][warn] Missing CSV for mission {k}: {csv_file}")
        continue
    line = load_mission_polylines(csv_file)
    mission_ids.append(k)
    mission_geoms.append(line)

gdf_lines = gpd.GeoDataFrame(
    {"mission": mission_ids, "geometry": mission_geoms},
    crs="EPSG:4326"
)

poly = wkt.loads(polygon_wkt)
poly_gdf = gpd.GeoDataFrame({"name": ["flight_zone"], "geometry": [poly]}, crs="EPSG:4326")

outside_pct = []
for k, row in gdf_lines.iterrows():
  pct = percent_length_outside(row.geometry, poly)
  outside_pct.append(pct)


viol_out = [(i, round(p, 4)) for i, p in enumerate(outside_pct) if p > 0.01]  # Tolerance
if viol_out:
    print("[Step5] Airspace violations (length % outside polygon):", viol_out)
else:
    print("[Step5] All mission polylines within polygon ✅")


# 5.5 Export a missions GeoJSON
gdf_lines.to_file(f"{out_dir}/missions.geojson", driver="GeoJSON")
print(f"[Step5] GeoJSON saved → {out_dir}/missions.geojson")


# TEMP CODE
# 5.6 Combined overview map
try:
    import contextily as cx
except Exception:
    cx = None

# Ensure 'mission' column exists for color mapping
if "mission" not in gdf_lines.columns:
    gdf_lines["mission"] = np.arange(len(gdf_lines))

lines_3857 = gdf_lines.to_crs(epsg=3857)
poly_3857 = poly_gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))
poly_3857.boundary.plot(ax=ax, linewidth=1.2, color="black", alpha=0.8, label="Flight Zone")

# color by mission id
lines_3857.plot(ax=ax, linewidth=2.0, alpha=0.9, column="mission", legend=False)

if cx:
    try:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    except Exception:
        pass

ax.set_title("Inspection Missions — Expanded Paths")
ax.set_axis_off()
plt.tight_layout()
fig.savefig(f"{out_dir}/missions_overview.png", dpi=180)
print(f"[Step5] Overview map saved → {out_dir}/missions_overview.png")

# 5.7 Save a concise validation report
report = {
    "num_missions": len(missions),
    "battery_ft": float(BATTERY_FT),
    "cap_ft_used_in_split": float(CAP_FT),
    "total_routing_distance_ft": float(sum(x["distance_ft"] for x in cap_report)),
    "max_mission_ft": float(max(x["distance_ft"] for x in cap_report)),
    "min_mission_ft": float(min(x["distance_ft"] for x in cap_report)),
    "all_within_cap": all(x["valid"] for x in cap_report),
    "coverage_ok": len(missed) == 0,
    "missed_targets": missed,
    "airspace_outside_pct_per_mission": [float(round(p, 6)) for p in outside_pct],
}
with open(f"{out_dir}/validation_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"[Step5] Validation report saved → {out_dir}/validation_report.json")

# === STEP 5B: Per-mission visualization ===
print("[Step5] Rendering per-mission maps…")

# Reproject for basemap & consistent layout
lines_3857 = gdf_lines.to_crs(epsg=3857)
poly_3857 = poly_gdf.to_crs(epsg=3857)

for idx, row in lines_3857.iterrows():
    mission_id = row["mission"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Mission {mission_id:02d}", fontsize=14)

    # Draw airspace boundary
    poly_3857.boundary.plot(ax=ax, linewidth=1.2, color="black", alpha=0.9)

    # Draw mission path line
    gpd.GeoSeries(row.geometry).plot(ax=ax,
                                     linewidth=3.0,
                                     alpha=0.9,
                                     color="tab:blue")

    # Try basemap
    if cx:
        try:
            cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
        except Exception:
            print(f"[warn] No basemap for mission {mission_id}")

    ax.set_axis_off()
    plt.tight_layout()

    mission_png = f"{out_dir}/mission_{mission_id:02d}.png"
    fig.savefig(mission_png, dpi=160)
    plt.close(fig)  # <-- prevent memory leaks/errors for many missions

    print(f" • Saved: {mission_png}")

print("[Step5] Per-mission maps complete!")

