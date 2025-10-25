import numpy as np
import pandas as pd
import os

# Detect sentinel used by SciPy's dijkstra predecessors (usually -9999)
def _detect_sentinel(P_row):
  vals = np.unique(P_row[:min(4096, P_row.shape[0])])
  negs = vals[vals < 0]
  return int(negs.min()) if negs.size else -9999

def get_waypoints_to_nav_map(N, P_cols, waypoints_mapping=None):
  if waypoints_mapping is not None:
    m = np.asarray(waypoints_mapping, dtype=int)
    assert m.shape[0] == N
    if np.any((m < 0) | (m >= P_cols)):
      raise ValueError("waypoints_mapping contains nav ids outside predecessor column range")
      return m
  if N <= P_cols:
    return np.arange(N, dtype=int)
  raise ValueError(f"Need waypoint→nav mapping (N_wp={N} > N_nav={P_cols}).")

def expand_leg_csgraph(i_wp, j_wp, P, wp2nav):
  i_nav = int(wp2nav[i_wp])
  j_nav = int(wp2nav[j_wp])
  if(i_nav == j_nav):
    return np.array([i_nav], dtype=int)

  row = P[i_wp]
  sentinel = _detect_sentinel(row)
  N_nav = P.shape[0]

  path = [j_nav]
  k = j_nav
  guard = 0
  while k != i_nav:
    pk = int(row[k])
    # If pk is sentinel or invalid, path is broken: fallback to direct step
    if pk == sentinel or pk < 0 or pk >= N_nav:
      return np.array([i_nav, j_nav], dtype=int)
    path.append(pk)

    k = pk
    guard += 1
    if guard > N_nav + 5: # cycle guard
      return np.array([i_nav, j_nav], dtype=int)

  path.reverse()
  return np.asarray(path, dtype=int)

def expand_route_csgraph(route_wp, P, wp2nav):
  out = []
  for a, b in zip(route_wp[:-1], route_wp[:-1]):
    leg = expand_leg_csgraph(a, b, P, wp2nav)
    if not out:
      out.extend(leg.tolist())
    else:
      out.extend(leg[1:].tolist() if out[-1] == leg[0] else leg.tolist())

  return np.asarray(out, dtype=int)

def export_expanded_coords(expanded_nav, coords_all, out_csv_path):
  mask = (expanded_nav >= 0) & (expanded_nav < coords_all.shape[0])
  if not np.all(mask):
      print(f"[warn] Dropped {np.count_nonzero(~mask)} expanded nodes not in coords range.")
  exp = expanded_nav[mask]
  xy = coords_all[exp]

  df = pd.DataFrame({"nav_id": exp, "lon": xy[:, 0], "lat": xy[:, 1], "order": np.arange(len(exp))})
  if out_csv_path:
      os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
      df.to_csv(out_csv_path, index=False)
      print(f"[Step3] Saved expanded path → {out_csv_path}")
  return df
