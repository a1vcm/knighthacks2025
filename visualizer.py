# visualizer.py — Plotly single-HTML overview with toggle (By Mission / By Drone)

import os
import json
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point, LineString, MultiPolygon
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- Simple helpers ----------
def load_mission_polylines(csv_file):
    df = pd.read_csv(csv_file)
    points = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    return LineString(points)

def percent_length_outside(line: LineString, polygon) -> float:
    if line.length == 0:
        return 0.0
    inside = line.intersection(polygon)
    inside_len = inside.length if not inside.is_empty else 0.0
    return max(0.0, 100.0 * (1.0 - inside_len / line.length))


# ---------- Internals ----------
def _mission_col(df):
    for c in ["Mission", "mission", "mission_id", "MissionId"]:
        if c in df.columns:
            return c
    return None


def _load_asset_points(points_path, asset_slice_path, photo_slice_path, mode="auto"):
    """Return (lons, lats, ids) for poles. Uses either direct assets or centroids of 4 photos."""
    pts = np.load(points_path)  # shape (K,2) [lon,lat]
    K = pts.shape[0]

    asset_lons, asset_lats, asset_ids = [], [], []

    asset_start = asset_end = None
    photo_start = photo_end = None
    if asset_slice_path and os.path.exists(asset_slice_path):
        sl = np.load(asset_slice_path)
        asset_start, asset_end = int(sl[0]), int(sl[-1])
    if photo_slice_path and os.path.exists(photo_slice_path):
        sl = np.load(photo_slice_path)
        photo_start, photo_end = int(sl[0]), int(sl[-1])

    def _direct_ok():
        return (asset_start is not None and asset_end is not None and
                0 <= asset_start < asset_end <= K)

    def _centroid_ok():
        return (photo_start is not None and photo_end is not None and
                0 <= photo_start < photo_end <= K and
                ((photo_end - photo_start) % 4 == 0))

    chosen = None
    if mode == "direct" and _direct_ok():
        chosen = "direct"
    elif mode == "centroid" and _centroid_ok():
        chosen = "centroid"
    elif mode == "auto":
        if _direct_ok():
            chosen = "direct"
        elif _centroid_ok():
            chosen = "centroid"

    if chosen == "direct":
        block = pts[asset_start:asset_end]
        asset_lons = block[:, 0].tolist()
        asset_lats = block[:, 1].tolist()
        asset_ids  = list(range(asset_end - asset_start))
        return asset_lons, asset_lats, asset_ids

    if chosen == "centroid":
        groups = (photo_end - photo_start) // 4
        if groups <= 0:
            return [], [], []
        for g in range(groups):
            idx0 = photo_start + 4*g
            block = pts[idx0:idx0+4]
            if block.shape[0] == 4:
                asset_lons.append(float(np.mean(block[:, 0])))
                asset_lats.append(float(np.mean(block[:, 1])))
                asset_ids.append(g)
        return asset_lons, asset_lats, asset_ids

    return [], [], []  # no valid way

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _polygon_to_rings(poly):
    if isinstance(poly, MultiPolygon):
        geoms = list(poly.geoms)
    else:
        geoms = [poly]
    rings = []
    for g in geoms:
        xs, ys = g.exterior.coords.xy
        xs, ys = list(xs), list(ys)
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs.append(xs[0]); ys.append(ys[0])
        rings.append((xs, ys))
    return rings

def _center_zoom(bounds):
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_c = (lon_min + lon_max) / 2.0
    lat_c = (lat_min + lat_max) / 2.0
    span = max(lon_max - lon_min, lat_max - lat_min, 1e-6)
    if span < 0.01: zoom = 14
    elif span < 0.05: zoom = 12
    elif span < 0.2: zoom = 10
    elif span < 1.0: zoom = 8
    elif span < 2.5: zoom = 6
    else: zoom = 4
    return lon_c, lat_c, zoom

def _compute_bounds(poly_rings, lines_lonlat):
    lons, lats = [], []
    for xs, ys in poly_rings:
        lons += xs; lats += ys
    for xs, ys in lines_lonlat:
        lons += xs; lats += ys
    if not lons:
        return (-98, 39, -97, 40)
    return (min(lons), min(lats), max(lons), max(lats))

def _get_style(style, token):
    # Default to a clean b/w basemap
    if token:
        return (style or "carto-positron", token)
    return ("carto-positron", None)


# ---------- Public builder with toggle ----------
def build_all_in_one_overview(
    polygon_wkt_path: str,
    missions_csv_dir: str,
    out_html: str = "out/missions_all_in_one_sidebyside.html",
    missions_summary_csv: str | None = None,
    schedule_csv: str | None = None,         # << keep passing the schedule file
    mapbox_token: str | None = None,
    map_style: str | None = None,            # None => default b/w
    line_width: float = 3.0,
    table_tint_alpha: float = 0.14,
    points_lonlat_npy: str | None = None,
    asset_indexes_npy: str | None = None,
    photo_indexes_npy: str | None = None,
    show_assets: bool = True,
    asset_mode: str = "auto",
    asset_marker_size: int = 6,
    asset_marker_color: str = "black",
):
    # ---------- (unchanged) load polygon, missions, metrics, depot, poles ----------
    with open(polygon_wkt_path, "r") as f:
        poly = wkt.loads(f.read().strip())
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly_rings = _polygon_to_rings(poly)

    summary_df = None
    if missions_summary_csv and os.path.exists(missions_summary_csv):
        try:
            summary_df = pd.read_csv(missions_summary_csv)
        except Exception:
            summary_df = None

    missions, lines_lonlat = [], []
    for name in sorted(os.listdir(missions_csv_dir)):
        if not (name.startswith("mission_") and name.endswith(".csv")):
            continue
        mid = int(name.split("_")[1].split(".")[0])
        csv_path = os.path.join(missions_csv_dir, name)
        line = load_mission_polylines(csv_path)
        xs, ys = list(line.coords.xy[0]), list(line.coords.xy[1])

        missions.append({
            "mission": mid, "csv": name, "line": line,
            "xs": xs, "ys": ys,
            "pct_outside": 0.0, "num_waypoints": "–",
            "distance_ft": None, "distance_mi": None, "Drone": None,
        })
        lines_lonlat.append((xs, ys))

    if summary_df is not None and len(missions):
        for m in missions:
            m["pct_outside"] = round(percent_length_outside(m["line"], poly), 2)
        s = summary_df.copy()
        s_cols = set(s.columns)
        for m in missions:
            row = s[s["mission"] == m["mission"]]
            if row.empty: continue
            r0 = row.iloc[0]
            if "num_waypoints" in s_cols: m["num_waypoints"] = int(r0["num_waypoints"])
            if "distance_ft"  in s_cols and pd.notna(r0["distance_ft"]): m["distance_ft"] = float(r0["distance_ft"])
            if "distance_mi"  in s_cols and pd.notna(r0["distance_mi"]): m["distance_mi"] = float(r0["distance_mi"])
            if "Drone"        in s_cols and pd.notna(r0["Drone"]):       m["Drone"] = str(r0["Drone"])

    missions.sort(key=lambda r: r["mission"])

    depot_lon = depot_lat = None
    if points_lonlat_npy and os.path.exists(points_lonlat_npy):
        pts = np.load(points_lonlat_npy)
        if pts.shape[0] > 0:
            depot_lon, depot_lat = float(pts[0, 0]), float(pts[0, 1])

    asset_lons = asset_lats = asset_ids = []
    if show_assets and points_lonlat_npy:
        asset_lons, asset_lats, asset_ids = _load_asset_points(
            points_lonlat_npy, asset_indexes_npy, photo_indexes_npy, mode=asset_mode
        )

    def _safe(vals): return [v for v in vals if isinstance(v, (int, float))]
    dist_fts = _safe([m["distance_ft"] for m in missions])
    waypoint_counts = _safe([m["num_waypoints"] if isinstance(m["num_waypoints"], (int, float)) else None for m in missions])

    # ---------- Metrics: By Mission ----------
    agg_mission = {
        "Total missions": len(missions),
        "Total distance (ft)": f"{sum(dist_fts):.1f}" if dist_fts else "–",
        "Total distance (mi)": f"{(sum(dist_fts)/5280.0):.2f}" if dist_fts else "–",
        "Average distance (ft)": f"{(np.mean(dist_fts)):.1f}" if dist_fts else "–",
        "Median distance (ft)": f"{(np.median(dist_fts)):.1f}" if dist_fts else "–",
        "Min / Max distance (ft)": (f"{min(dist_fts):.1f} / {max(dist_fts):.1f}") if dist_fts else "–",
        "Total waypoints": f"{int(sum(waypoint_counts))}" if waypoint_counts else "–",
    }
    metrics_cols = ["Metric", "Value"]
    metrics_vals_mission = list(agg_mission.items())

    # ---------- Metrics: By Drone (if schedule is available) ----------
    # --- Metrics: By Drone (if schedule is available) ---
    metrics_vals_drone = None
    sched = None
    if schedule_csv and os.path.exists(schedule_csv):
        sched = pd.read_csv(schedule_csv)

    mcol = _mission_col(sched)
    if mcol is not None and "Drone" in sched.columns:
        # map: mission id -> distance (from missions_summary)
        dist_map = {int(m["mission"]): float(m["distance_ft"] or 0.0) for m in missions}

        # robust mission-id extraction: accept "05", "5", "Mission 05", etc.
        s_ids = (
            sched[mcol]
            .astype(str)
            .str.extract(r'(\d+)')[0]               # keep only the digits
            .astype('Int64')                        # nullable int
        )
        tmp = sched.copy()
        tmp["_mid_"] = s_ids
        tmp = tmp.dropna(subset=["_mid_"]).copy()
        tmp["_mid_"] = tmp["_mid_"].astype(int)

        per_drone_dist = tmp.groupby("Drone")["_mid_"].apply(
            lambda s: float(sum(dist_map.get(int(mid), 0.0) for mid in s))
        )

        drones = list(per_drone_dist.index)
        total_drone_dist = float(per_drone_dist.sum())
        avg_drone_dist = float(per_drone_dist.mean()) if len(per_drone_dist) else 0.0

        makespan = "–"
        if {"Start_s", "End_s"}.issubset(set(sched.columns)):
            makespan_s = float(sched["End_s"].max() - sched["Start_s"].min())
            hh = int(makespan_s // 3600); mm = int((makespan_s % 3600) // 60)
            makespan = f"{hh:02d}:{mm:02d} (hh:mm)"

        breakdown = ", ".join([f"D{d}: {per_drone_dist[d]:.0f} ft" for d in drones])
        metrics_vals_drone = [
            ("Drones", len(drones)),
            ("Total distance (ft)", f"{total_drone_dist:.1f}"),
            ("Avg distance per drone (ft)", f"{avg_drone_dist:.1f}"),
            ("Makespan", makespan),
            ("Per-drone distance", breakdown if breakdown else "–"),
        ]


    # ---------- Map layout ----------
    style, token = _get_style(map_style, mapbox_token)
    bounds = _compute_bounds([], lines_lonlat)
    lon_c, lat_c, zoom = _center_zoom(bounds)
    zoom = max(zoom + 1, 3)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.50, 0.50],
        specs=[
            [{"type": "mapbox"}, {"type": "table"}],
            [{"type": "xy"},      {"type": "table"}],
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
        row_heights=[0.60, 0.40],
    )

    palette_mission = go.Figure().layout.template.layout.colorway or [
        "#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A",
        "#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"
    ]
    palette_drone = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Flight Zone
    for xs, ys in poly_rings:
        fig.add_trace(
            go.Scattermapbox(
                lon=xs, lat=ys, mode="lines",
                fill="toself", fillcolor="rgba(0,0,0,0.05)",
                line=dict(width=2, color="#444"),
                name="Flight Zone",
                hoverinfo="skip",
                visible=True,
            ),
            row=1, col=1
        )

    # Poles (always on)
    if show_assets and asset_lons and asset_lats:
        fig.add_trace(
            go.Scattermapbox(
                lon=asset_lons, lat=asset_lats, mode="markers",
                marker=dict(size=asset_marker_size, color=asset_marker_color),
                name="Poles",
                text=[f"Pole {i:04d}" for i in asset_ids],
                hovertemplate="<b>%{text}</b><br>Lon %{lon:.6f}<br>Lat %{lat:.6f}<extra></extra>",
                visible=True,
            ),
            row=1, col=1
        )

    # Depot (always on)
    if depot_lon is not None and depot_lat is not None:
        fig.add_trace(
            go.Scattermapbox(
                lon=[depot_lon], lat=[depot_lat],
                mode="markers",
                marker=dict(size=10, color="black", symbol="star"),
                name="Depot",
                hovertemplate="<b>Depot</b><br>%{lon:.6f}, %{lat:.6f}<extra></extra>",
                visible=True,
            ),
            row=1, col=1
        )

    # Mission traces (default visible)
    mission_trace_indices = []
    for i, rec in enumerate(missions):
        color = palette_mission[i % len(palette_mission)]
        hover = f"<b>Mission {rec['mission']:02d}</b>"
        if isinstance(rec.get("distance_ft"), (int, float)):
            hover += f"<br>Distance: {rec['distance_ft']:.1f} ft"
        if isinstance(rec.get("distance_mi"), (int, float)):
            hover += f" ({rec['distance_mi']:.2f} mi)"
        hover += f"<br># Waypoints: {rec['num_waypoints']}"
        if rec.get("Drone"): hover += f"<br>Drone: {rec['Drone']}"
        hover += f"<br>% Outside: {rec['pct_outside']}"

        fig.add_trace(
            go.Scattermapbox(
                lon=rec["xs"], lat=rec["ys"],
                mode="lines",
                line=dict(width=line_width, color=color),
                name=f"Mission {rec['mission']:02d}",
                hovertemplate=hover + "<extra></extra>",
                visible=True,
                legendgroup="missions",
            ),
            row=1, col=1
        )
        mission_trace_indices.append(len(fig.data)-1)

    # Drone traces (start hidden)
    # Drone traces (start hidden)
    drone_trace_indices = []
    if sched is not None:
        mcol = _mission_col(sched)
        if mcol is not None and "Drone" in sched.columns:
        # robust mission-id extraction again
          s_ids = (
              sched[mcol]
              .astype(str)
              .str.extract(r'(\d+)')[0]
              .astype('Int64')
          )
          tmp = sched.copy()
          tmp["_mid_"] = s_ids
          tmp = tmp.dropna(subset=["_mid_"]).copy()
          tmp["_mid_"] = tmp["_mid_"].astype(int)

          mission_to_xy = {int(m["mission"]): (m["xs"], m["ys"]) for m in missions}
          for d_i, (drone_id, grp) in enumerate(tmp.groupby("Drone")):
              if "Start_s" in grp.columns:
                  grp = grp.sort_values("Start_s")
              xs_all, ys_all = [], []
              for _, row in grp.iterrows():
                  mid = int(row["_mid_"])
                  if mid in mission_to_xy:
                      xs, ys = mission_to_xy[mid]
                      if xs_all: xs_all.append(None); ys_all.append(None)
                      xs_all += xs; ys_all += ys
              if xs_all:
                  fig.add_trace(
                      go.Scattermapbox(
                          lon=xs_all, lat=ys_all, mode="lines",
                          line=dict(width=line_width+1, color=palette_drone[d_i % len(palette_drone)]),
                          name=f"Drone {drone_id}",
                          hovertemplate=f"<b>Drone {drone_id}</b><extra></extra>",
                          visible=False,
                          legendgroup="drones",
                      ),
                      row=1, col=1
                  )
                  drone_trace_indices.append(len(fig.data)-1)


    # ----- Metrics tables (two traces in same cell; toggle visibility) -----
    # Mission metrics (visible)
    metrics_table_mission_idx = None
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="#f0f0f0", align="left"),
            cells=dict(
                values=[[k for k, _ in metrics_vals_mission], [v for _, v in metrics_vals_mission]],
                align="left",
            ),
            visible=True,
        ),
        row=1, col=2
    )
    metrics_table_mission_idx = len(fig.data)-1

    # Drone metrics (hidden if not available)
    metrics_table_drone_idx = None
    if metrics_vals_drone is not None:
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"], fill_color="#f0f0f0", align="left"),
                cells=dict(
                    values=[[k for k, _ in metrics_vals_drone], [v for _, v in metrics_vals_drone]],
                    align="left",
                ),
                visible=False,
            ),
            row=1, col=2
        )
        metrics_table_drone_idx = len(fig.data)-1

    # Distances bar (no legend entry so bottom legend stays clean)
    bar_x, bar_y, bar_colors = [], [], []
    for i, rec in enumerate(missions):
        if isinstance(rec.get("distance_ft"), (int, float)):
            bar_x.append(f"{rec['mission']:02d}")
            bar_y.append(rec["distance_ft"])
            bar_colors.append(palette_mission[i % len(palette_mission)])
    if bar_x:
        fig.add_trace(
           go.Bar(
              x=bar_x, y=bar_y, marker_color=bar_colors, name="Distance (ft)",
              hovertemplate="Mission %{x}<br>%{y:.1f} ft<extra></extra>",
            ),
            row=2, col=1
        )

        fig.update_xaxes(fixedrange=True, row=2, col=1)
        fig.update_yaxes(title_text="Distance (ft)", fixedrange=True, row=2, col=1)

    # Mission table (bottom-right) — unchanged
    cols = ["Mission", "# Waypoints", "Distance (ft)", "Distance (mi)", "% Outside", "Drone", "File"]
    rows, row_colors = [], []
    for i, rec in enumerate(missions):
        rows.append([
            f"{rec['mission']:02d}",
            rec["num_waypoints"],
            f"{rec['distance_ft']:.1f}" if isinstance(rec.get("distance_ft"), (int,float)) else "–",
            f"{rec['distance_mi']:.2f}" if isinstance(rec.get("distance_mi"), (int,float)) else "–",
            f"{rec['pct_outside']}",
            rec.get("Drone", "—"),
            rec["csv"],
        ])
        row_colors.append(_hex_to_rgba(palette_mission[i % len(palette_mission)], table_tint_alpha))
    if rows:
        nrows, ncols = len(rows), len(cols)
        fill_matrix = [[row_colors[r]] * ncols for r in range(nrows)]
        fig.add_trace(
            go.Table(
                header=dict(values=cols, align="left", fill_color="#f0f0f0"),
                cells=dict(values=list(map(list, zip(*rows))), align="left", fill_color=fill_matrix),
            ),
            row=2, col=2
        )

    # ---------- Layout (legend at bottom, flat) ----------
    fig.update_layout(
        mapbox=dict(
            style=style,
            accesstoken=token,
            center=dict(lon=lon_c, lat=lat_c),
            zoom=zoom,
        ),
        title="Inspection Missions — Overview (Interactive)",
        hovermode="closest",
        uirevision="keep",
        legend=dict(
            orientation="h",
            y=-0.18, x=0.5, xanchor="center", yanchor="top",   # << flat bottom legend
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)", borderwidth=1
        ),
        margin=dict(l=10, r=10, t=60, b=120),
    )

    # ---------- Toggle buttons (moved up so they don't cover the table) ----------
    # Build the visibility arrays for the two modes
    def vis_for(mode: str):
        """mode in {'mission','drone'} — returns a list of booleans, one per trace."""
        visible = []
        for idx, tr in enumerate(fig.data):
            name = (tr.name or "")
            is_map = hasattr(tr, "lon")
            is_zone  = (name == "Flight Zone")
            is_poles = (name == "Poles")
            is_depot = (name == "Depot")
            is_mission = (getattr(tr, "legendgroup", None) == "missions")
            is_drone   = (getattr(tr, "legendgroup", None) == "drones")
            # Metrics tables (by index)
            is_mission_metrics = (idx == metrics_table_mission_idx)
            is_drone_metrics   = (metrics_table_drone_idx is not None and idx == metrics_table_drone_idx)

            if is_zone or is_poles or is_depot or not is_map:
                # Always keep non-map traces (tables, bar) and static overlays visible
                # …except the metrics table pair, which we toggle below
                keep = True
            else:
                keep = (mode == "mission" and is_mission) or (mode == "drone" and is_drone)

            # Override for the two metrics tables
            if is_mission_metrics:
                keep = (mode == "mission")
            if is_drone_metrics:
                keep = (mode == "drone")

            visible.append(bool(keep))
        return visible

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            x=0.99, y=1.14, xanchor="right", yanchor="top",   # << moved above the table
            direction="right",
            showactive=True,
            buttons=[
                dict(label="By Mission", method="update", args=[{"visible": vis_for("mission")}]),
                dict(label="By Drone",   method="update", args=[{"visible": vis_for("drone")}]),
            ],
        )]
    )

    # ---------- Write HTML ----------
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["zoomInMapbox","zoomOutMapbox","resetViewMapbox"],
        },
    )
    print(f"[viz] Saved → {out_html}")
    return out_html
