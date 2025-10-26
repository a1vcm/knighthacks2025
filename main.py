# main.py — cleaned flow with 3-drone scheduling before visualization

# Std/3rd-party
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString

# Project imports
from data_metrics import display_data_metrics                 # optional
from tsp_solver import ortools_single_tsp, route_distance
from predecessor_formatter import (
    get_waypoints_to_nav_map, expand_route_csgraph, export_expanded_coords
)
from heuristics import solve_vrp_distance_cap                 # (greedy_split available but unused here)
from visualizer import build_all_in_one_overview, load_mission_polylines, percent_length_outside
from schedule import schedule_missions

# ---------------------------
# Load data
# ---------------------------
data_dir = "data"
out_dir = "out"
Path(out_dir).mkdir(parents=True, exist_ok=True)

data = {}
data["assets"]       = np.load(f"{data_dir}/asset_indexes.npy")
data["distances"]    = np.load(f"{data_dir}/distance_matrix.npy")
data["photo"]        = np.load(f"{data_dir}/photo_indexes.npy")
data["points"]       = np.load(f"{data_dir}/points_lat_long.npy")
data["predecessors"] = np.load(f"{data_dir}/predecessors.npy")
data["waypoints"]    = np.load(f"{data_dir}/waypoint_indexes.npy")

with open(f"{data_dir}/polygon_lon_lat.wkt") as f:
    polygon_wkt = f.read().strip()

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")  # optional

# Optional: quick metrics
# display_data_metrics(data)

# Shorthands
D = data["distances"]
P = data["predecessors"]
coords = data["points"]
N = D.shape[0]
depot = 0

# ---------------------------
# Step 2 — TSP seed (complete tour)
# ---------------------------
tsp_route = ortools_single_tsp(D, depot=depot)
dist_ft = route_distance(tsp_route, D)
print(f"TSP route length: {len(tsp_route)} nodes")
print(f"Total tour distance: {dist_ft:.1f} ft ({dist_ft/5280.0:.2f} mi)")
print(f"Route starts at {tsp_route[0]}, ends at {tsp_route[-1]}")

# ---------------------------
# Step 3 — Expand abstract route via predecessors (single CSV for demo)
# ---------------------------
N_wp = D.shape[0]
N_nav = P.shape[1]
wp2nav = get_waypoints_to_nav_map(N_wp, N_nav, waypoints_mapping=None)

expanded_nav_tour = expand_route_csgraph(tsp_route, P, wp2nav)
export_expanded_coords(expanded_nav_tour, coords, out_csv_path=f"{out_dir}/tsp_expanded_path.csv")
print(f"[Step3] Expanded nav-node path length: {len(expanded_nav_tour)}")

# ---------------------------
# Step 4 — VRP with battery cap → missions
# ---------------------------
BATTERY_FT = 37725.0
SAFETY_BUFFER = 0.99
CAP_FT = BATTERY_FT * SAFETY_BUFFER

missions = solve_vrp_distance_cap(
    D,
    depot=depot,
    cap_ft=CAP_FT,
    time_limit_s=60,
    vehicle_upper_bound=None,
    vehicle_fixed_cost=None,
    force_visit=True,   # every non-depot node must be visited
    slack_ratio=1.0,
    meta="GLS",
)

used_nodes = sum(len(r) - 2 for r in missions)  # minus depot at both ends
print(f"[VRP] Missions created: {len(missions)}, visited nodes: {used_nodes}/{D.shape[0]-1} (cap {CAP_FT:.1f} ft)")
print(f"[Step4] Missions created: {len(missions)} (cap {CAP_FT:.1f} ft incl. buffer {SAFETY_BUFFER*100:.0f}%)\n")

# Expand/export each mission to CSV and collect summary rows
mission_metrics = []
for k, route in enumerate(missions):
    # Expand to nav nodes and export
    nav_path = expand_route_csgraph(route, P, wp2nav)
    csv_path = f"{out_dir}/mission_{k:02d}.csv"
    export_expanded_coords(nav_path, coords, out_csv_path=csv_path)

    # Distance on abstract D
    m_ft = route_distance(route, D)
    mission_metrics.append({
        "mission": k,
        "num_waypoints": max(0, len(route) - 2),
        "distance_ft": float(m_ft),
        "distance_mi": float(m_ft / 5280.0),
        "csv": Path(csv_path).name,
    })

# Save baseline summary
summary_df = pd.DataFrame(mission_metrics)
summary_df.to_csv(f"{out_dir}/missions_summary.csv", index=False)
print("[Step4] Summary preview:")
print(summary_df.head(min(10, len(summary_df))))
print(f"[Step4] Saved → {out_dir}/missions_summary.csv")

# ---------------------------
# Multi-drone scheduling (3 drones, minimize finish time)
# ---------------------------
summary_df = pd.read_csv(f"{out_dir}/missions_summary.csv")  # read what we just wrote

speed_ft_s = 25.0        # ft/s -> ~17 mph
turnaround_s = 15 * 60   # 15 min pad per mission

schedule_df, sched_metrics = schedule_missions(
    summary_df,
    num_drones=3,
    speed_ft_s=speed_ft_s,
    turnaround_s=turnaround_s,
    launch_stagger_s=0.0,
    respect_priority=True,
    out_csv=f"{out_dir}/drone_schedule.csv",
    out_html=f"None",
)

# Merge assigned Drone into missions_summary for visualization
m2d = schedule_df[["mission", "Drone"]].drop_duplicates("mission")
summary_df = summary_df.merge(m2d, on="mission", how="left")
summary_df.to_csv(f"{out_dir}/missions_summary.csv", index=False)
print(f"[Schedule] Saved: {out_dir}/drone_schedule.csv and {out_dir}/drone_schedule.html")

# ---------------------------
# Step 5 — Coverage/battery checks + GeoJSON + Interactive Overview
# ---------------------------
# Coverage
all_visited = set()
for r in missions:
    all_visited.update([n for n in r if n != depot])
photo_targets = np.arange(0, D.shape[0], dtype=int)
photo_targets = photo_targets[photo_targets != depot]
missed = sorted(set(photo_targets) - all_visited)

print("\n[Step5] Coverage Check")
print(f"Visited: {len(all_visited)} / {len(photo_targets)}")
print("Missed targets:", "NONE" if not missed else missed)

# Battery check
cap_report = []
violations = []
for k, r in enumerate(missions):
    m_ft = route_distance(r, D)
    ok = m_ft <= CAP_FT + 1e-6
    cap_report.append({"mission": k, "distance_ft": m_ft, "valid": ok, "slack_ft": CAP_FT - m_ft})
    if not ok:
        violations.append((k, m_ft))
if violations:
    print("[Step5] Battery violations found:", violations)
else:
    print("[Step5] All missions within cap!")

# Lines for GeoJSON & map-overlay checks
mission_ids, mission_geoms = [], []
for k in range(len(missions)):
    csv_file = f"{out_dir}/mission_{k:02d}.csv"
    if not Path(csv_file).exists():
        print(f"[Step5][warn] Missing CSV for mission {k}: {csv_file}")
        continue
    line = load_mission_polylines(csv_file)
    mission_ids.append(k)
    mission_geoms.append(line)

gdf_lines = gpd.GeoDataFrame({"mission": mission_ids, "geometry": mission_geoms}, crs="EPSG:4326")

poly = wkt.loads(polygon_wkt)
poly_gdf = gpd.GeoDataFrame({"name": ["flight_zone"], "geometry": [poly]}, crs="EPSG:4326")

outside_pct = []
for _, row in gdf_lines.iterrows():
    pct = percent_length_outside(row.geometry, poly)
    outside_pct.append(pct)

viol_out = [(i, round(p, 4)) for i, p in enumerate(outside_pct) if p > 0.01]
if viol_out:
    print("[Step5] Airspace violations (length % outside polygon):", viol_out)
else:
    print("[Step5] All mission polylines within polygon ✅")

# GeoJSON export
gdf_lines.to_file(f"{out_dir}/missions.geojson", driver="GeoJSON")
print(f"[Step5] GeoJSON saved → {out_dir}/missions.geojson")

# ---------------------------
# Plotly Overview (built AFTER scheduling so Drone column is present)
# ---------------------------
html_path = build_all_in_one_overview(
    polygon_wkt_path=f"{data_dir}/polygon_lon_lat.wkt",
    missions_csv_dir=out_dir,
    missions_summary_csv=f"{out_dir}/missions_summary.csv",
    schedule_csv=f"{out_dir}/drone_schedule.csv",   # <--- NEW
    out_html=f"{out_dir}/missions_all_in_one_sidebyside.html",
    mapbox_token=MAPBOX_TOKEN,
    map_style="carto-positron",                     # crisp b/w
    line_width=3.0,
    points_lonlat_npy=f"{data_dir}/points_lat_long.npy",
    asset_indexes_npy=f"{data_dir}/asset_indexes.npy",
    photo_indexes_npy=f"{data_dir}/photo_indexes.npy",
    show_assets=True,
    asset_mode="centroid",
    asset_marker_size=6,
    asset_marker_color="black",
)

# Final overall stats
total_ft = float(summary_df["distance_ft"].sum()) if len(summary_df) else 0.0
print(f"\n[Step5] Total routing distance across missions: {total_ft:.1f} ft ({total_ft/5280.0:.2f} mi)")
print(f"[Done] Overview HTML → {html_path}")
