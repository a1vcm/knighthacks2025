import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

def _fmt_hms(seconds):
  s = max(0, int(round(float(seconds))))
  h = s // 3600; s -= h * 3600
  m = s // 60; s -= m * 60
  return f"{h:02d}:{m:02d}:{s:02d}"

def schedule_missions(
  mission_summary,
  num_drones,
  speed_ft_s,
  turnaround_s,
  launch_stagger_s,
  respect_priority,
  out_csv,
  out_html
):
  df = mission_summary.copy()
  if "mission" not in df.columns or "distance_ft" not in df.columns:
    raise ValueError("missions_summary.csv must have columns: mission, distance_ft")

  if "priority" not in df.columns:
    df["priority"] = 0

  # Compute the flight times
  df["flight_time_s"] = df["distance_ft"].astype(float) / float(speed_ft_s)

  # Sort: priority
  sort_cols = ["priority", "flight_time_s"] if respect_priority else ["flight_time_s"]
  df = df.sort_values(by=sort_cols, ascending=[False, False] if respect_priority else [False])

  # Drone state
  available_at = [i * float(launch_stagger_s) for i in range(num_drones)]
  schedules = [[] for _ in range(num_drones)]

  for _, row in df.iterrows():
    mid = int(row["mission"])
    flight_s = float(row["flight_time_s"])
    prio = int(row.get("priority", 0))

    d = min(range(num_drones), key=lambda i: available_at[i])
    start = available_at[d]
    finish = start + flight_s

    schedules[d].append({
      "Drone": f"Drone {d+1}",
      "mission": mid,
      "Priority": prio,
      "Start_s": start,
      "Finish_s": finish,
      "Duration_s": flight_s,
    })

    available_at[d] = finish + float(turnaround_s)

  # Flatten and Decorate
  rows = []
  for d_sched in schedules:
    for it in d_sched:
      rows.append({
        **it,
        "Mission": f"M{it['mission']:02d}",
        "Start_hms": _fmt_hms(it["Start_s"]),
        "Finish_hms": _fmt_hms(it["Finish_s"]),
        "Duration_hms": _fmt_hms(it["Duration_s"]),
      })

  schedule_df = pd.DataFrame(rows).sort_values(by=["Drone", "Start_s"]).reset_index(drop=True)
  makespan_s = float(schedule_df["Finish_s"].max()) if len(schedule_df) else 0.0
  util_sec = schedule_df.groupby("Drone")["Duration_s"].sum().to_dict()
  util_pct = {k: (v / makespan_s * 100.0 if makespan_s > 0 else 0.0) for k, v in util_sec.items()}

  # write CSV
  Path("out").mkdir(parents=True, exist_ok=True)
  schedule_df.to_csv(out_csv, index=False)

  fig = px.timeline(
    schedule_df,
    x_start="Start_s", x_end="Finish_s",
    y="Drone", color="Mission",
    hover_data=["Priority", "Start_hms", "Finish_hms", "Duration_hms"],
    title=f"Multi-Drone Schedule (Drones={num_drones}) — Makespan {_fmt_hms(makespan_s)}",
  )

  fig.update_yaxes(autorange="reversed")
  fig.update_layout(
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    xaxis_title="Time (seconds)",
  )
  fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)

  metrics = {
    "makespan_s": makespan_s,
    "makespan_hms": _fmt_hms(makespan_s),
    "utilization_pct": util_pct,
  }

  print(f"[Schedule] Drones={num_drones}  Makespan={metrics['makespan_hms']}")
  for d, pct in sorted(util_pct.items()):
    print(f"  {d}: Utilization {pct:.1f}% (flight only)")

  return schedule_df, metrics

