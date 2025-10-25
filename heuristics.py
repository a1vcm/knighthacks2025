
def greedy_split_by_battery(full_route, D, battery_ft, depot=0):
  """
  Split a closed tour into missions that each start/end at depot and stay <= battery_ft.
  Greedy rule: only add next stop if we can still afford to return to depot.
  """
  visits = [v for v in full_route if v != depot]
  missions = []
  i = 0
  N = len(visits)

  while i < N:
    mission = [depot]
    cur = depot
    dist = 0.0
    progressed = False

    while i < N:
      nxt = visits[i]
      cost_go = D[cur, nxt]
      cost_back = D[nxt, depot]
      # ensure we can still return to depot after adding nxt
      if dist + cost_go + cost_back <= battery_ft:
        mission.append(nxt)
        dist += cost_go
        cur = nxt
        i += 1
        progressed = True
      else:
          break

    # If we couldn't add any visit, try to fit a single-visit mission
    if not progressed:
      nxt = visits[i]
      cost_roundtrip = D[depot, nxt] + D[nxt, depot]
      if cost_roundtrip > battery_ft:
        # This means the cap is too low for this stop at all; fail loudly.
        raise ValueError(
            f"Waypoint {nxt} cannot fit in a single mission under cap {battery_ft:.1f} ft "
            f"(needs {cost_roundtrip:.1f} ft)."
        )
      mission.append(nxt)
      i += 1

    mission.append(depot)
    missions.append(mission)

  return missions
