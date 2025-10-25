# visualizer.py — Plotly single-HTML overview with legend + tables + bar chart
import os
import json
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- Existing helpers (kept) ----------
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
        # ensure closed ring for mapbox fill
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
    if token:
        return (style or "outdoors", token)
    return ("open-street-map", None)


# ---------- Public builder ----------
def build_all_in_one_overview(
    polygon_wkt_path: str,
    missions_csv_dir: str,
    out_html: str = "out/missions_all_in_one_sidebyside.html",
    missions_summary_csv: str | None = None,
    mapbox_token: str | None = None,
    map_style: str | None = None,
    line_width: float = 3.0,
    table_tint_alpha: float = 0.14,
):
    """
    Builds a single interactive HTML with:
      - Map (left): shaded flight polygon + all mission polylines (color-coded)
      - Top-right table: aggregate metrics
      - Bottom-left: mission distance bar chart (axes locked; scroll goes to map)
      - Bottom-right: color-tinted mission table (colors match lines)
    """
    # --- Polygon ---
    with open(polygon_wkt_path, "r") as f:
        poly = wkt.loads(f.read().strip())
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly_rings = _polygon_to_rings(poly)

    # --- Load missions (+ optional summary stats) ---
    summary_df = None
    if missions_summary_csv and os.path.exists(missions_summary_csv):
        try:
            summary_df = pd.read_csv(missions_summary_csv)
        except Exception:
            summary_df = None

    missions = []
    lines_lonlat = []
    for name in sorted(os.listdir(missions_csv_dir)):
        if not (name.startswith("mission_") and name.endswith(".csv")):
            continue
        mid = int(name.split("_")[1].split(".")[0])
        csv_path = os.path.join(missions_csv_dir, name)
        line = load_mission_polylines(csv_path)
        xs, ys = list(line.coords.xy[0]), list(line.coords.xy[1])

        rec = {
            "mission": mid,
            "csv": name,
            "line": line,
            "xs": xs,
            "ys": ys,
            "pct_outside": round(percent_length_outside(line, poly), 2),
            "num_waypoints": "–",
            "distance_ft": None,
            "distance_mi": None,
        }
        if summary_df is not None:
            row = summary_df[summary_df["mission"] == mid]
            if not row.empty:
                rec["num_waypoints"] = int(row.iloc[0].get("num_waypoints", "–"))
                if not pd.isna(row.iloc[0].get("distance_ft", np.nan)):
                    rec["distance_ft"] = float(row.iloc[0]["distance_ft"])
                if not pd.isna(row.iloc[0].get("distance_mi", np.nan)):
                    rec["distance_mi"] = float(row.iloc[0]["distance_mi"])
                # If you later add photos_visited/assets_visited, surface them here:
                if "photos_visited" in row.columns and not pd.isna(row.iloc[0].get("photos_visited", np.nan)):
                    rec["photos_visited"] = int(row.iloc[0]["photos_visited"])
                if "assets_visited" in row.columns and not pd.isna(row.iloc[0].get("assets_visited", np.nan)):
                    rec["assets_visited"] = int(row.iloc[0]["assets_visited"])

        missions.append(rec)
        lines_lonlat.append((xs, ys))

    missions.sort(key=lambda r: r["mission"])

    # --- Aggregates for metrics table & bar chart ---
    def _safe(vals):
        return [v for v in vals if isinstance(v, (int, float))]
    dist_fts = _safe([m["distance_ft"] for m in missions])
    waypoint_counts = _safe([m["num_waypoints"] if isinstance(m["num_waypoints"], (int, float)) else None for m in missions])

    agg = {
        "Total missions": len(missions),
        "Total distance (ft)": f"{sum(dist_fts):.1f}" if dist_fts else "–",
        "Total distance (mi)": f"{(sum(dist_fts)/5280.0):.2f}" if dist_fts else "–",
        "Average distance (ft)": f"{(np.mean(dist_fts)):.1f}" if dist_fts else "–",
        "Median distance (ft)": f"{(np.median(dist_fts)):.1f}" if dist_fts else "–",
        "Min / Max distance (ft)": (f"{min(dist_fts):.1f} / {max(dist_fts):.1f}") if dist_fts else "–",
        "Total waypoints": f"{int(sum(waypoint_counts))}" if waypoint_counts else "–",
    }

    # --- Map view & style ---
    style, token = _get_style(map_style, mapbox_token)

    # Focus the initial view on the mission lines (tighter than polygon bounds)
    bounds = _compute_bounds([], lines_lonlat)
    lon_c, lat_c, zoom = _center_zoom(bounds)
    zoom = max(zoom + 1, 3)  # small bias to zoom tighter

    # --- Layout grid: 2 rows x 2 cols (map left 50%, tables right 50%, bar chart under map) ---
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.50, 0.50],   # 50/50 split
        specs=[
            [{"type": "mapbox"}, {"type": "table"}],
            [{"type": "xy"},      {"type": "table"}],   # bar chart left; mission table right
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
        row_heights=[0.60, 0.40],
    )

    # --- Color palette ---
    palette = go.Figure().layout.template.layout.colorway or [
        "#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A",
        "#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"
    ]

    # --- Shaded polygon (fill) + outline on map (Row1, Col1) ---
    # (We still draw the polygon, even though we center on missions)
    with_fill = True
    for xs, ys in poly_rings:
        fig.add_trace(
            go.Scattermapbox(
                lon=xs, lat=ys, mode="lines",
                fill="toself" if with_fill else None,
                fillcolor="rgba(0,128,0,0.12)" if with_fill else None,
                line=dict(width=2, color="green"),
                name="Flight Zone",
                hoverinfo="skip",
            ),
            row=1, col=1
        )

    # --- Mission lines on map ---
    for i, rec in enumerate(missions):
        color = palette[i % len(palette)]
        hover = f"<b>Mission {rec['mission']:02d}</b>"
        if isinstance(rec.get("distance_ft"), (int, float)):
            hover += f"<br>Distance: {rec['distance_ft']:.1f} ft"
        if isinstance(rec.get("distance_mi"), (int, float)):
            hover += f" ({rec['distance_mi']:.2f} mi)"
        hover += f"<br># Waypoints: {rec['num_waypoints']}"
        if "photos_visited" in rec: hover += f"<br>Photos: {rec['photos_visited']}"
        if "assets_visited" in rec: hover += f"<br>Poles: {rec['assets_visited']}"
        hover += f"<br>% Outside: {rec['pct_outside']}"

        fig.add_trace(
            go.Scattermapbox(
                lon=rec["xs"], lat=rec["ys"],
                mode="lines",
                line=dict(width=line_width, color=color),
                name=f"Mission {rec['mission']:02d}",
                hovertemplate=hover + "<extra></extra>",
                uirevision="keep",   # preserve map state on redraws
            ),
            row=1, col=1
        )

    # --- Metrics table (Row1, Col2) ---
    metrics_cols = ["Metric", "Value"]
    metrics_vals = list(agg.items())
    fig.add_trace(
        go.Table(
            header=dict(values=metrics_cols, fill_color="#f0f0f0", align="left"),
            cells=dict(
                values=[[k for k, _ in metrics_vals], [v for _, v in metrics_vals]],
                align="left",
            ),
        ),
        row=1, col=2
    )

    # --- Bar chart of distances (Row2, Col1). Lock axes so scroll goes to map. ---
    bar_x, bar_y, bar_colors = [], [], []
    for i, rec in enumerate(missions):
        if isinstance(rec.get("distance_ft"), (int, float)):
            bar_x.append(f"{rec['mission']:02d}")
            bar_y.append(rec["distance_ft"])
            bar_colors.append(palette[i % len(palette)])

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

    # --- Color-coded mission table (Row2, Col2) ---
    # If you added photos_visited / assets_visited to missions_summary.csv, they’ll appear here automatically.
    cols = ["Mission", "# Waypoints", "Distance (ft)", "Distance (mi)", "% Outside", "File"]
    # If available, prepend Photos/Poles columns:
    if any("photos_visited" in m for m in missions) or any("assets_visited" in m for m in missions):
        cols = ["Mission", "# Waypoints", "Photos", "Poles", "Distance (ft)", "Distance (mi)", "% Outside", "File"]

    rows = []
    row_colors = []
    for i, rec in enumerate(missions):
        base = [
            f"{rec['mission']:02d}",
            rec["num_waypoints"],
        ]
        if "Photos" in cols:
            base += [
                rec.get("photos_visited", "–"),
                rec.get("assets_visited", "–"),
            ]
        base += [
            f"{rec['distance_ft']:.1f}" if isinstance(rec.get("distance_ft"), (int, float)) else "–",
            f"{rec['distance_mi']:.2f}" if isinstance(rec.get("distance_mi"), (int, float)) else "–",
            f"{rec['pct_outside']}",
            rec["csv"],
        ]
        rows.append(base)
        row_colors.append(_hex_to_rgba(palette[i % len(palette)], table_tint_alpha))

    if rows:
        nrows = len(rows)
        ncols = len(cols)
        fill_matrix = [[row_colors[r]] * ncols for r in range(nrows)]
        fig.add_trace(
            go.Table(
                header=dict(values=cols, align="left", fill_color="#f0f0f0"),
                cells=dict(
                    values=list(map(list, zip(*rows))),
                    align="left",
                    fill_color=fill_matrix,
                ),
            ),
            row=2, col=2
        )

    # --- Layout & map config ---
    fig.update_layout(
        mapbox=dict(
            style=style,
            accesstoken=mapbox_token,  # token may be None for "open-street-map"
            center=dict(lon=lon_c, lat=lat_c),
            zoom=zoom,
        ),
        title="Inspection Missions — Overview (Interactive)",
        hovermode="closest",
        uirevision="keep",
        legend=dict(
            orientation="h",
            y=-0.12, x=0.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)", borderwidth=1
        ),
        margin=dict(l=10, r=10, t=50, b=80),
    )

    # --- Write HTML once with proper interaction config ---
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,  # wheel zoom on the map
            "displaylogo": False,
            # Explicit Mapbox buttons:
            "modeBarButtonsToAdd": ["zoomInMapbox", "zoomOutMapbox", "resetViewMapbox"],
            # (Optional) If you want to remove 2D cartesian zoom buttons:
            # "modeBarButtonsToRemove": ["zoomIn2d", "zoomOut2d"],
        },
    )
    print(f"[viz] Saved → {out_html}")
    return out_html
