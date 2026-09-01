import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
from pyomo.environ import *
from cost_models import get_price_vector
from tqdm import tqdm


def get_optimal_welfare(game_dict):
    """ Computes the joint strategy that maximizes the cumulative welfare of all buyers.
    
    Notes: If we contrain the supply to be less than total demand or don't consider value of final position - opt welfare is 0
           If we constraint the supply to be less than toal demand, then opt welfare is 0, even with final position utiliyt
           If we don't constrain supply and consider final poisiton utility, it is beneficial for the supplier to over-supply and drive down cost for the
                 the buyer, knowing that at the end, it can build that position back cheaper since their temp impact beta is less than the beta faced by the buyers.
    """
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    supply = game_dict["supply"]
    
    # Objective function
    def objective(demand):
        demand.shape = (n, T)
        pts, _ = get_price_vector(game_dict, demand)
        total_utility = 0
        # TODO: We can vectorize these below if runtime becomes an issue here
        if reserve:
            total_utility = np.sum([reserve[i]*np.sum(demand[i]) - np.dot(pts, demand[i]) for i in range(n)])
        else:
            total_utility = -1*np.sum([np.dot(pts, demand[i]) for i in range(n)])
            
        # scipy default is to minimize - hence the negative
        return -1*total_utility
    
    # Constraint: sum of demand == Vs[i] when no reserve; demand <= Vs[i] with reserve
    cons = []
    for i in range(n):
        cons.append({
            'type': 'eq' if not reserve else 'ineq',
            'fun': lambda demand, i=i: Vs[i] - np.sum(demand[i*T:(i+1)*T])
        })
    
    # No per-round bounds
    bounds = [(None, None) for _ in range(n*T)]

    x0 = np.ones(n*T)
    if game_dict["exp"] == 1:
        result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    else:
        result = scipy_minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=cons)   
    
    # If we are not using reserve, then return the positive cost (lower better)
    # Otherwise, return the utility (higher better)
    return result.x.reshape((n,T)), result.fun if not reserve else -1*result.fun
    
        

def get_buyer_best_response(game_dict, trader_strat, i):
    T = game_dict["T"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    supply = game_dict["supply"]

    # Objective function
    def objective(demand):
        pts, _ = get_price_vector(game_dict, trader_strat, i, demand)
    
        total_utility = 0
        if reserve:
            total_utility = np.sum(demand)*reserve[i] - np.dot(pts, demand)
        else:
            total_utility = -1*np.dot(pts, demand) 
            
        # scipy default is to minimize - hence the negative
        return -1*total_utility

    # Constraint: sum of demand == Vs[i]
    #cons = None
    cons = ({
        'type': 'eq' if not reserve else 'ineq',
        'fun': lambda demand: Vs[i] - np.sum(demand)
    })

    # Bounds: demand >= 0
    bounds = [(None, None) for _ in range(T)]

    # Initial guess: split Vs[i] evenly
    x0 = np.ones(T) * (Vs[i] / T)
    if game_dict["exp"] == 1:
        result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    else:
        result = scipy_minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=cons)

    if not result.success:
        print("Scipy failed:", result.message)
        return None
    else:
        return result.x


def get_cost(game_dict, demand_matrix):
    supply = game_dict["supply"]
    price_vector, perm_price_vector = get_price_vector(game_dict, demand_matrix)
    total_costs = []
    for i in range(game_dict["n"]):
        cost = np.dot(price_vector, demand_matrix[i])
        total_costs.append(cost) 
    return price_vector, perm_price_vector, total_costs
   

def verify_equilibrium(game_dict, demand_matrix, supply):
    """ Verify whether a given set of strategies at at an equilibrium
    """
    n, T, Vs = game_dict["n"], game_dict["T"], game_dict["Vs"]
    supply = game_dict["supply"]

    price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix, supply)
    print(f"The total cost of current strategy is: {total_cost}")

    # check the best response for each of the buyers
    eps = 0.01
    for i in range(n):
        br_demand_i = get_buyer_best_response(game_dict, demand_matrix, supply, i)
        step_size = np.linalg.norm(br_demand_i - demand_matrix[i]) 
        if step_size >= eps:
            print(f"Agent {i} best responds with: {br_demand_i}")
            return False

    return True


def find_equilibrium_br(game_dict, verbose=True, get_welfare=True):
    # Initialize supply
    n, T, Vs = game_dict["n"], game_dict["T"], game_dict["Vs"]
    reserve = game_dict["reserve"]
    supply = game_dict["supply"]

    # create an initial demand matrix where everyone buys the whole order up-front.
    demand_matrix = np.zeros((n, T))
    for i in range(n):
        demand_matrix[i, 0] = Vs[i]
    
    # Now try and find a Nash Equilibrium through best-response play.
    iter, eps, max_iter = 0, 0.00001, 1000
    while True:
        update = False
        step_sizes = []

        for i in range(n):
            br_demand_i = get_buyer_best_response(game_dict, demand_matrix, i)
            step_size = np.linalg.norm(br_demand_i - demand_matrix[i]) 
            if step_size >= eps:
                demand_matrix[i] = br_demand_i
                update = True
            step_sizes.append(step_size)

        iter += 1
        print(f"Iter: {iter} with the largest step size being: {max(step_sizes)}") if verbose else None
        
        if not update:
            found = True
            print(f"Found Equilibrium in {iter} iterations") if verbose else None
            break
        if iter >= 1000:
            found = False
            break

    if found:
        price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix)
        revenue = np.dot(price_vector, supply)

        print(f"The equilibrium demand matrix is: {demand_matrix}") if verbose else None
        print(f"This leads to realized price {price_vector} and perm prices: {perm_price_vector}") if verbose else None
        print(f"The cost to each trader is: {total_cost}") if verbose else None
        print(f"The total cost is: {sum(total_cost)}") if verbose else None

        ppoa = None
        if get_welfare:
            if reserve:
                eq_welfare = np.sum([reserve[i]*np.sum(demand_matrix[i]) - np.dot(price_vector, demand_matrix[i]) for i in range(n)])
            else:
                eq_welfare = sum(total_cost)
            demand_welf, opt_welfare = get_optimal_welfare(game_dict)
            price_opt_welfare, perm_price_opt, _ = get_cost(game_dict, demand_welf) 
            if reserve:
                print(f"\n\n The utility (higher better) of Equilibrium is: {eq_welfare}")
                print(f"The optimal cum utility is: {opt_welfare} with demand: {demand_welf} and prices: {price_opt_welfare}")
                ppoa = np.abs(opt_welfare)/np.abs(eq_welfare)
                print(f"The PPoA is: {ppoa}")
            else:
                print(f"\n\n The total cost (lower better) of Equilibrium is: {eq_welfare}")
                print(f"The optimal cum cost is: {opt_welfare} with demand: {demand_welf} and prices: {price_opt_welfare}")
    else:
        print(f"Equilibrium not found in {max_iter} iterations.")
        print(f"Demand matrix: {demand_matrix}")
        print(f"Supply vector: {supply}")

    return found, demand_matrix, ppoa


def check_random_equilibrium(n, T, alpha, beta):
    num_iters = 100
    for i in tqdm(range(num_iters)):
        Vs = np.random.randint(0, 20, n)
        supply = np.random.rand(0, 20//T, T)
        game_dict = {
            "n" :   n,
            "T" :   T,
            "p_0" : 0,
            "Vs" : Vs,
            "alpha" : alpha/T,
            "beta" : beta,
            "supply" : supply,
            "reserve" : None,
            "exp" : 1
        }
        found, _, _ = find_equilibrium_br(game_dict, supply_player=True, verbose=False)
        supply_eq, demand_eq, opt_welfare = get_optimal_welfare(game_dict)
        
        if not found:
            print(Vs)
            exit(0)


def best_response_test():
    n, T, alpha, beta = 2, 2, 1, 1
    Vs = [10 for i in range(n)]
    supply = [0, 0, 0]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "beta" : beta,
        "p_0" : 0,
        "Vs" : Vs,
        "alpha" : alpha/T,
        "supply" : supply,
        "reserve" : None,
        "exp" : 1
    }
    
    demand_matrix = np.array([
        [5, 5],
        [1, 9]
    ])
    best_response = get_buyer_best_response(game_dict, demand_matrix, 0)
    print(f"Player 0 best response is: {best_response}")

def price_of_anarchy_equality():
    n, T, alpha, beta = 3, 3, 1, 0
    eps = 0.01
    supply = [0 for i in range(T)]
    V = 10
    x = V/T

    Vs = [V, 2*V, 3*V]
    p_0 = x

    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : p_0,
        "Vs" : Vs,
        "alpha" : alpha,
        "beta" : beta,
        "supply" : supply,
        "reserve" : None,
        "exp" : 1
    }

    # Compute the equilibrium using optimizers
    _, demand_matrix, ppoa = find_equilibrium_br(game_dict, get_welfare=False, verbose=False)
    equi_price, _ = get_price_vector(game_dict, demand_matrix)
    eq_cost = np.sum([np.dot(equi_price, demand_matrix[i]) for i in range(n)]) 
    print(f"The equilibrium cost is: {eq_cost} with demand {demand_matrix} and prices: {equi_price}")

    demand_welf, opt_welfare = get_optimal_welfare(game_dict)
    print(f"The optimal computed cost: {opt_welfare} with demand: {demand_welf}")

    ppoa = eq_cost/opt_welfare
    print(f"The price of anarchy is {ppoa}")

def price_of_anarchy():
    n, T, alpha, beta = 2, 2, 1, 0
    eps = 0.01
    supply = [0 for i in range(T)]
    V = 10
    x = V/T
    reserve = [x, x-eps] 

    Vs = [V, V]
    p_0 = x

    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : p_0,
        "Vs" : Vs,
        "alpha" : alpha,
        "beta" : beta,
        "supply" : supply,
        "reserve" : reserve,
        "exp" : 1
    }

    # Compute the equilibrium using optimizers
    _, demand_matrix, ppoa = find_equilibrium_br(game_dict, get_welfare=False, verbose=False)
    equi_price, _ = get_price_vector(game_dict, demand_matrix)
    eq_welfare = np.sum([reserve[i]*np.sum(demand_matrix[i]) - np.dot(equi_price, demand_matrix[i]) for i in range(n)]) 
    print(f"The equilibrium welfare is: {eq_welfare} with demand {demand_matrix} and prices: {equi_price}")

    # Compute the equilibrium algebraiacally
    M = np.array([
        [2, 1, 1, 0],
        [1, 2, 1, 1],
        [1, 0, 2, 1],
        [1, 1, 1, 2]
    ])
    v = np.array([0, 0, -eps, -eps])
    all_demand = np.matmul(np.linalg.inv(M), v) * (1/alpha)
    all_demand = all_demand.reshape((2,2))
    print(np.linalg.inv(M))
    print(f"The algebraically computed equilibrium demand is {demand_matrix}") 

    demand_welf, opt_welfare = get_optimal_welfare(game_dict)
    print(f"The optimal computed utility: {opt_welfare} with demand: {demand_welf}")

    delta = eps/(3*alpha)
    opt_welfare_alg = 2*x*eps + 2*(eps**2)/(3*alpha) - (eps**2)/3*alpha
    print(f"The algebraically computed opt utility: {opt_welfare_alg}")

    print(f"The PPoA is: {opt_welfare_alg/eq_welfare}")

if __name__ == "__main__":
    n, T, alpha, beta = 2, 5, 5, 1
    Vs = [10, 20]
    reserve = [1000 for i in range(n)]
    supply = [0 for i in range(T)]

    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 2.0,
        "Vs" : Vs,
        "alpha" : alpha,
        "beta" : beta,
        "supply" : supply,
        "reserve" : reserve,
        "exp" : 1
    }
        
    _, demand_matrix, ppoa = find_equilibrium_br(game_dict, get_welfare=False, verbose=True)
    print(demand_matrix)

    
    