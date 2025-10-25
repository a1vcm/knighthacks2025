import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

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


def _estimate_vehicle_upper_bound(D: np.ndarray, depot: int, cap_ft: float) -> int:
    # crude upper bound so the model has room; capped by N-1
    N = D.shape[0]
    roundtrips = D[depot, 1:] + D[1:, depot]
    lb = int(np.ceil(roundtrips.sum() / max(cap_ft, 1.0)))
    return int(max(1, min(N - 1, lb + 6)))

def solve_vrp_distance_cap(
    D,
    depot=0,
    cap_ft=37725.0,
    time_limit_s=60,
    vehicle_upper_bound=None,
    vehicle_fixed_cost=None,
    force_visit=True,          # True = all non-depot must be visited
    slack_ratio=1.0,
    meta="GLS",
    log_search=False,
):
    N = int(D.shape[0])

    # Guard: each node must at least fit a roundtrip
    cap_eff = float(cap_ft) * float(slack_ratio)
    for i in range(1, N):
        rt = float(D[depot, i] + D[i, depot])
        if rt > cap_eff + 1e-6:
            raise ValueError(
                f"Waypoint {i} needs {rt:.1f} ft roundtrip, exceeds cap {cap_eff:.1f} ft."
            )

    # Defaults
    if vehicle_upper_bound is None:
        vehicle_upper_bound = _estimate_vehicle_upper_bound(D, depot, cap_ft)
    vehicle_upper_bound = int(max(1, vehicle_upper_bound))

    if vehicle_fixed_cost is None:
        vehicle_fixed_cost = int(round(float(cap_ft) * 2.0))
    else:
        vehicle_fixed_cost = int(vehicle_fixed_cost)

    cap_int = int(round(float(cap_ft) * float(slack_ratio)))

    # OR-Tools setup
    manager = pywrapcp.RoutingIndexManager(N, vehicle_upper_bound, int(depot))
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(round(float(D[a, b])))

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # Distance (battery) dimension
    routing.AddDimension(transit, 0, cap_int, True, "Distance")

    # Discourage extra vehicles
    for v in range(vehicle_upper_bound):
        routing.SetFixedCostOfVehicle(int(vehicle_fixed_cost), int(v))

    # Only add disjunctions when drops are allowed
    if not force_visit:
        BIG = int(1e9)
        for node in range(N):
            if node == depot:
                continue
            routing.AddDisjunction([manager.NodeToIndex(node)], BIG)

    # Search params
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        if meta.upper() == "GLS"
        else routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    )
    params.time_limit.FromSeconds(int(time_limit_s))
    params.log_search = bool(log_search)

    sol = routing.SolveWithParameters(params)
    if not sol:
        raise RuntimeError(
            "VRP infeasible or time too low; increase vehicle_upper_bound/time_limit_s, "
            "keep slack_ratio=1.0, or temporarily set force_visit=False to diagnose."
        )

    # Extract routes as [0, ..., 0]
    routes = []
    for v in range(vehicle_upper_bound):
        idx = routing.Start(v)
        nxt = sol.Value(routing.NextVar(idx))
        if routing.IsEnd(nxt):
            continue  # unused vehicle
        route = []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        route.append(int(depot))
        if len(route) > 2:
            routes.append(route)

    # Safety
    if not routes:
        raise RuntimeError("Solver found a solution but no vehicles were used—check disjunctions/penalties.")

    return routes
