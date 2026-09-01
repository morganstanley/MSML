import numpy as np
import matplotlib.pyplot as plt

from main import get_optimal_welfare, find_equilibrium_br, get_buyer_best_response
from algorithms import extra_gradient_equilibrium
from cost_models import get_price_vector


def generalized_ppoa(alpha=1, beta=1, eps=0.01):
    n, T, alpha = 2, 2, 1
    supply = [0 for i in range(T)]
    
    # Algebraically pre-compute the equilibrium strategy.
    # Note that the equilibrium strategy here does NOT depend on x or V. The cost, however, does
    Q = np.ones((T, T))*alpha + (alpha+2*beta)*np.eye(T)
    A = np.tril(alpha*np.ones((T, T))) + beta*np.eye(T)
    M = np.block([
        [Q, A],
        [A, Q]
    ])
    z = np.array([0 for i in range(T)] + [-eps for i in range(T)])
    alg_equi_demand = np.matmul(np.linalg.inv(M), z)
    alg_equi_demand = alg_equi_demand.reshape((2,T))
    max_val = np.max(np.sum(np.abs(alg_equi_demand), axis=1))
    
    # Choose any constant x. Results should hold regardless. We will choose V accordingly
    # For T > 2, the algebraic bound here will be a lower bound on the true optimal welfare.
    x = 10
    reserve = [x, x-eps]
    delta = eps/(3*alpha + 2*beta)
    V = max(T*x + T*delta, max_val)
    alg_opt_welfare = 2*eps*x + 2*(eps**2)/(3*alpha + 2*beta) - (eps**2)/(3*alpha + 2*beta)
    
    # Instantiate the game instance
    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : x,
        "Vs" : [V for i in range(n)],
        "alpha" : alpha,
        "beta" : beta,
        "supply" : supply,
        "reserve" : reserve,
        "exp" : 1
    }

    alg_equi_price, _ = get_price_vector(game_dict, alg_equi_demand)
    alg_equi_welfare = np.abs(np.sum([reserve[i]*np.sum(alg_equi_demand[i]) - np.dot(alg_equi_price, alg_equi_demand[i]) for i in range(n)]))
    alg_ppoa = alg_opt_welfare/alg_equi_welfare

    print(f"The algebraic demand matrix is: {alg_equi_demand}")
    print(f"The price vector is: {alg_equi_price}")
    print(f"The algebraically computed equilibrium welfare is {alg_equi_welfare}")
    print(f"The algebraically computed optimal welfare is {alg_opt_welfare}")
    print(f"The algebraic PPoA is: {alg_ppoa}")

    # Now verify this by actually creating the game instance and computing the welfare/equi
    equi_demand = extra_gradient_equilibrium(game_dict, eps=eps**3)
    equi_price, _ = get_price_vector(game_dict, equi_demand)
    equi_welfare = np.sum([reserve[i]*np.sum(equi_demand[i]) - np.dot(equi_price, equi_demand[i]) for i in range(n)]) 
    equi_welfare = np.abs(equi_welfare)

    demand_welf, opt_welfare = get_optimal_welfare(game_dict)
    opt_price, _ = get_price_vector(game_dict, demand_welf)
    opt_welfare = np.abs(opt_welfare)
    ppoa = opt_welfare/equi_welfare

    br_0 = get_buyer_best_response(game_dict, demand_welf, 0)
    br_1 = get_buyer_best_response(game_dict, demand_welf, 1)

    print(f"The optimal demand matrix is: {demand_welf}")
    print(f"Buyer 1 best-response at optimal: {br_0}")
    print(f"Buyer 2 best-response at optimal: {br_1}")
    print(f"The optimal price: {opt_price}")
    print(f"The exact equilibrium welfare is: {equi_welfare}")
    print(f"The exact computed optimal welfare is {opt_welfare}")
    print(f"The exact PPoA is: {ppoa}")
    

if __name__ == "__main__":
    generalized_ppoa(alpha=1, beta=1, eps=0.01)
    


