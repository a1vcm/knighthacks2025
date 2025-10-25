import numpy as np
import pandas as pd


def display_data_metrics(data):
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
