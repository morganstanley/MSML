from pyomo.environ import *
import numpy as np

def get_seller_best_response(game_dict, trader_strat):
    """ the utility is the revenue made in selling shares minus the cost need to rebuild that 
        position at the last price
    """
    print(trader_strat)
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]

    # Objective function
    def objective(supply):
        pts = get_price_vector(game_dict, trader_strat, supply)
        revenue = np.dot(pts, supply)
        
        last_step_walrus = pts[-1] - beta*(np.sum(trader_strat[:,T-1]) - supply[T-1])
        #return -1*revenue + np.sum(supply)*last_step_walrus + 0.5*np.sum(supply)**2
        return -1*revenue 
    
    # Constraint: sum of demand == Vs[i]
    cons = ({
        'type': 'ineq',
        'fun': lambda demand: np.sum(Vs) - np.sum(demand)
    })

    # Bounds: demand >= 0
    bounds = [(0, None) for _ in range(T)]

    # Initial guess: split Vs[i] evenly
    x0 = np.ones(T) * (Vs[0] / T)
    result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=None)

    if not result.success:
        print("Scipy failed:", result.message)
        return None
    else:
        return result.x

def get_buyer_best_response_pyomo(game_dict, trader_strat, supply_strat, i, symmetric=False):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta,  p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]

    model = ConcreteModel()
    model.T = RangeSet(0, T-1)

    # Variables
    model.demand = Var(model.T, domain=NonNegativeReals)

    # Demand sum constraint
    def demand_sum_rule(m):
        return sum(m.demand[t] for t in m.T) == Vs[i]
    model.demand_sum_con = Constraint(rule=demand_sum_rule)

    # Objective
    def obj_rule(m):
        obj = 0
        for t in m.T:
            if symmetric:
                obj += p_0 * m.demand[t]
                for l in range(t+1):
                    obj += n*alpha*m.demand[t]*m.demand[l]
                obj -= alpha*m.demand[t]*np.sum(supply_strat[:t+1]) 
            else:
                # both approaches are the same. Used for sanity checks
                # pt = p_0
                # for l in range(t+1):
                #     pt += alpha*(sum(trader_strat[:, l]) - trader_strat[i,l] + m.demand[l] - supply_strat[l])
                # pt += beta*(sum(trader_strat[:, t]) - trader_strat[i, l] + m.demand[t])
                # obj += m.demand[t] * pt

                obj += p_0 * m.demand[t]
                for l in range(t+1):
                    obj += alpha * m.demand[t] * m.demand[l]
                    remaining_demand = sum(trader_strat[:, l]) - trader_strat[i, l]
                    obj += alpha * m.demand[t] * (remaining_demand - supply_strat[l])
                obj += beta * m.demand[t] * m.demand[t]
                obj += beta * m.demand[t] * (sum(trader_strat[:, t]) - trader_strat[i, t] - supply_strat[t])
        return obj
    model.obj = Objective(rule=obj_rule, sense=minimize)

    # Solve
    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=False)

    if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
        print("Pyomo failed with status:", result.solver.status)
        return None
    else:
        demand_val = np.array([value(model.demand[t]) for t in model.T])
        return demand_val


def get_seller_best_response_pyomo(game_dict, trader_strat):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]

    model = ConcreteModel()
    model.T = RangeSet(0, T-1)

    # Variables
    model.supply = Var(model.T, domain=NonNegativeReals)

    #print(f"Demand in seller best response: {trader_strat}")
    #print(f"Game dict in seller br: {game_dict}")
    # Objective
    def obj_rule(m):
        revenue = 0
        for t in m.T:
            # compute the price p_t
            pt = p_0
            for l in range(t+1):
                pt += alpha*(np.sum(trader_strat[:, l]) - m.supply[l])
            pt += beta*np.sum(trader_strat[:, t])
            revenue += pt * m.supply[t]
        return revenue
        
    model.obj = Objective(rule=obj_rule, sense=maximize)

    # Solve
    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=False)
    if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
        print("Pyomo failed with status:", result.solver.status)
        return None
    else:
        demand_val = np.array([value(model.supply[t]) for t in model.T])
        revenue = value(model.obj)
        return demand_val
    
    

def equilibrium_test_symmetric():
    n = 2
    game_dict = {
        "n" :   n,
        "T" :   3,
        "p_0" : 0,
        "Vs" : [10 for i in range(n)],
        "alpha" : 1
    }
    supply = [5, 5]
    
    symmetric_opt = get_buyer_best_response(
        game_dict, 
        np.zeros(shape=(game_dict["n"], game_dict["T"])), 
        supply, 
        0, 
        symmetric=True
    ) 
    symmetric_demand = np.array([symmetric_opt for i in range(n)])
    price_vector, total_cost = get_cost(game_dict, symmetric_demand, supply)
    print(f"The symmetric_demand matrix is: {symmetric_demand}")
    print(f"Symmetric strat leads to price: {price_vector} and total cost: {np.sum(total_cost)} to all traders") 

    # Now try and find a Nash Equilibrium through best-response play. The starting symmetric
    # position is just an intial point.
    j, demand_matrix = 0, symmetric_demand
    while True:
        update = False
        for i in range(n):
            br_demand_i = get_trader_best_response_scipy(game_dict, demand_matrix, supply, i)
            if np.linalg.norm(br_demand_i - demand_matrix[i]) >= 0.01:
                demand_matrix[i] = br_demand_i
                update = True
        j += 1
        print(f"Completed round: {j}")
        if not update:
            print("Found Equilibrium")
            break

    price_vector, total_cost = get_cost(game_dict, demand_matrix, supply)
    print(f"The equilibrium demand matrix is: {demand_matrix}")
    print(f"This leads to price {price_vector} giving total costs: {np.sum(total_cost)} to traders")