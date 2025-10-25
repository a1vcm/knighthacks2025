from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def route_distance(route, D):
    return float(sum(D[a, b] for a, b in zip(route[:-1], route[1:])))

def ortools_single_tsp(D, depot=0, time_limit_s=20):
    """
    Solve a single-vehicle TSP over all nodes in D, starting/ending at `depot`.
    Returns a list of node indices [0, ..., 0].
    """
    N = D.shape[0]
    manager = pywrapcp.RoutingIndexManager(N, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(round(D[a, b]))   # OR-Tools requires integer costs
    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    sol = routing.SolveWithParameters(params)
    if not sol:
        raise RuntimeError("OR-Tools TSP failed to find a solution")

    # Extract route
    route = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    route.append(depot)
    return route
