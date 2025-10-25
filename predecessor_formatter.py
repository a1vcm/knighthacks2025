import numpy as np
import pandas as pd
import os

def _detect_sentinel(P_row: np.ndarray) -> int:
    vals = np.unique(P_row[: min(4096, P_row.shape[0])])
    negs = vals[vals < 0]
    return int(negs.min()) if negs.size else -9999

def get_waypoints_to_nav_map(N: int, P_cols: int, waypoints_mapping=None):
    """
    Map waypoint ids (rows of D/P) to nav-node ids (columns of P).
    If you have an explicit mapping array (length N), pass it. Else assume identity when N <= P_cols.
    """
    if waypoints_mapping is not None:
        m = np.asarray(waypoints_mapping, dtype=int)
        assert m.shape[0] == N, "waypoints_mapping must have length N"
        if np.any((m < 0) | (m >= P_cols)):
            raise ValueError("waypoints_mapping contains nav ids outside predecessor column range")
        return m  # <-- ensure we return the provided mapping

    if N <= P_cols:
        return np.arange(N, dtype=int)

    raise ValueError(f"Need waypoint→nav mapping (N_wp={N} > N_nav={P_cols}).")

def expand_leg_csgraph(i_wp: int, j_wp: int, P: np.ndarray, wp2nav: np.ndarray) -> np.ndarray:
    i_nav = int(wp2nav[int(i_wp)])
    j_nav = int(wp2nav[int(j_wp)])
    if i_nav == j_nav:
        return np.array([i_nav], dtype=int)

    row = P[int(i_wp)]
    sentinel = _detect_sentinel(row)
    N_nav = P.shape[1]  # <-- columns are nav nodes (fixed)

    path = [j_nav]
    k = j_nav
    guard = 0
    while k != i_nav:
        pk = int(row[k])
        if pk == sentinel or pk < 0 or pk >= N_nav:
            # fallback: draw direct link (this is what produced straight lines before)
            return np.array([i_nav, j_nav], dtype=int)
        path.append(pk)
        k = pk
        guard += 1
        if guard > N_nav + 5:
            return np.array([i_nav, j_nav], dtype=int)

    path.reverse()
    return np.asarray(path, dtype=int)

def expand_route_csgraph(route_wp, P: np.ndarray, wp2nav: np.ndarray) -> np.ndarray:
    # normalize to list[int]
    if isinstance(route_wp, (int, np.integer)):
        route_wp = [int(route_wp)]
    else:
        route_wp = [int(x) for x in list(route_wp)]

    if len(route_wp) == 0:
        return np.array([], dtype=int)
    if len(route_wp) == 1:
        return np.array([int(wp2nav[route_wp[0]])], dtype=int)

    out = []
    for a, b in zip(route_wp[:-1], route_wp[1:]):  
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
