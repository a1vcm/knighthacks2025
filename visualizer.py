import pandas as pd
from shapely.geometry import Point, LineString

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
