import numpy as np
import osqp
from scipy.sparse import csc_matrix
import sys

import time

def get_linear_operator(game_dict):
    """ Recall that our equilibrium can be expressed as a variational inequality
    with a linear operator Mx + c. This operator matrix depends only on the instance params
    alpha and b depends on supply, reserve, and initial price. 
    
    For large instances, these are large and they should only be computed once and stored in mem.
    """
    n, T, alpha, beta = game_dict["n"], game_dict["T"], game_dict["alpha"], game_dict["beta"] 
    p_0 = game_dict["p_0"]
    reserve = np.array(game_dict["reserve"])
    supply = np.array(game_dict["supply"])

    # construct the M matrix
    Q = np.diag([alpha + 2*beta for i in range(T)]) + alpha*np.ones((T,T))
    A = alpha*np.tril(np.ones((T, T))) + np.diag([beta for i in range(T)])
    I_n = np.diag([1 for i in range(n)])
    J_n = np.ones((n,n))
    M = np.kron(J_n, A) - np.kron(I_n, A) + np.kron(I_n, Q)

    # construct the c vector
    assert reserve is not None
    B = -alpha*np.tril(np.ones((T, T))) - beta*np.eye(T)
    supp = np.matmul(B, supply)
    supp_kron = np.kron(np.ones(n), supp)
    r_p = [p_0 - reserve[i] for i in range(n)]
    r_p_kron = np.kron(r_p, np.ones(T)) 
    c = supp_kron + r_p_kron
    return M, c


def get_linear_operator_bayesian(game_dict):
    """ Recall that our equilibrium can be expressed as a variational inequality
    with a linear operator Mx + c (for linear utilities). For the bayesian setting, 
    our operator matrix M is an nkT x nkT matrix. Similarly, c is an nkT vector
    
    For large instances, these are large and they should only be computed once and stored in mem.
    """

    # types is an n x k dimensional matrix
    n, T, k, p_0 = game_dict["n"], game_dict["T"], game_dict["k"], game_dict["p_0"]
    
    # reserves/Vs is a mapping from (player i, type l) to reserve/Vs
    # these are all the expected values since that is all that matters due to linearity
    reserves = game_dict["reserves"]
    Vs = game_dict["Vs"]
    
    # For the alphas and betas, we need a mapping from joint_type to alpha beta
    alphas = game_dict["alphas"]
    betas = game_dict["betas"]

    # This is the joint type distribution. For n=2, this is a k1 x k2 dim matrix
    type_dist = game_dict["type_dist"]
    marginals = [np.sum(type_dist, axis=1), np.sum(type_dist, axis=0)]

    #print(f"Betas: {betas}")
    #print(f"Type dist: {type_dist}")
    #print(f"Marginals: {marginals}")
    #print(f"reserves: {reserves}")

    # We assume supply to be independent of type. So we just care about the EV of the supply vector    
    supply = np.array(game_dict["supply"])

    # construct the M matrix, which can be thought of as an n x n block matrix.
    M = np.zeros((n*k*T, n*k*T))
    for i in range(n):
        for j in range(n):
            # The diagonals
            if i == j:
                for l in range(k):
                    marginal_types = [(l, other_l) for other_l in range(k)]
                    # Assuming uniform type distribution here
                    alpha = sum([(1/k) * alphas[joint_type] for joint_type in marginal_types])
                    beta = sum([(1/k) * betas[joint_type] for joint_type in marginal_types])
                    Q = np.diag([alpha + 2*beta for i in range(T)]) + alpha*np.ones((T,T))
                    prob = marginals[i][l] 
                    scaled_Q = prob*Q

                    # scaled_Q is a TxT matrix. I want to build a kT x kT matrix where each diagonal is one of these scaled Q and the rest are 0.
                    start_row = i*(k * T) + l * T
                    end_row = i *(k * T) + (l + 1) * T
                    start_col = j*(k * T) + (l*T)
                    end_col = j * (k * T) + (l + 1) * T
                    M[start_row:end_row, start_col:end_col] = scaled_Q
                    #print(f"i={i}, j={j}, l={l}: prob:{prob}, Q:\n{Q}")
            
            # The off diagonals
            else:
                for l_i in range(k):
                    for l_j in range(k):
                        alpha, beta = alphas[(l_i, l_j)], betas[(l_i, l_j)]
                        A = alpha*np.tril(np.ones((T, T))) + np.diag([beta for i in range(T)])                        
                        prob = type_dist[l_i][l_j] 
                        scaled_A = prob * A

                        start_row = i*(k * T) + l_i * T
                        end_row = i *(k * T) + (l_i + 1) * T
                        start_col = j*(k * T) + (l_j * T)
                        end_col = j * (k * T) + (l_j + 1) * T
                        M[start_row:end_row, start_col:end_col] = scaled_A
                        #print(f"i={i}, j={j}, l_i={l_i}, l_j:{l_j}: prob:{prob}, A:\n{A}")
    
    # construct the c vector
    # We assume supply is the same for all types - i.e. no distribution
    B = -alpha*np.tril(np.ones((T, T))) - beta*np.eye(T)
    supp = np.matmul(B, supply)
    supp_kron = np.kron(np.ones(n*k), supp)
    
    r_p = []
    for i in range(n):
        for l in range(k):
            r_p.append(p_0 - reserves[i,l])
    c = np.kron(r_p, np.ones(T)) + supp_kron
    return M, c


def project_feasible_analytical(game_dict, z):
    # Assume constraints of the form -V <= \sum{hit} <= V
    n, T = game_dict["n"], game_dict["T"]
    Vs = np.array(game_dict["Vs"])
    H = z.reshape((n,T))
 
    offset_minus = np.maximum(np.sum(H, axis=1) - Vs, np.zeros(n)) / T
    H -= offset_minus[:, None]
    offset_plus = np.maximum(-1*Vs - np.sum(H, axis=1), np.zeros(n)) / T
    H += offset_plus[:, None]
    return H.flatten()


def project_feasible_analytical_bayesian(game_dict, z):
    n, k, T = game_dict["n"], game_dict["k"], game_dict["T"]
    Vs = game_dict["Vs"]
    H = z.reshape((n*k,T))
    flat_Vs = Vs.reshape(-1)

    offset_minus = np.maximum(np.sum(H, axis=1) - flat_Vs, np.zeros(n*k)) / T
    H -= offset_minus[:, None]
    offset_plus = np.maximum(-1*flat_Vs - np.sum(H, axis=1), np.zeros(n*k)) / T
    H += offset_plus[:, None]

    return H.flatten()


def project_feasible(game_dict, z):
    """ Given z, which is an nT dim array of demands, project this to the feasibly region.
    Note that the feasible region must be convex, which means this projection is only well-defined
    if we have inequality constraints. For equality constraints, we can solve directly using matrix
    inverse, or with the inequality constraints but with very large reserve prices.

    Projection under l2 distance is generally quadratic program: we want to project z to a convex region
    minimize 1/2||x - z||_2^2 = minimize 1/2 x^T I x - z^T x. The constraints can be expressed as l <= Ax <= c
    In our case, we want projection such that each T sized vector sums to less than V_i, with no lower bound 
    contraints for now. We will used OSQP, which is an extremely fast solver specifically for quadratic programs.
    
    IMPORTANT: USE THIS FUNCTION IF YOU HAVE CONSTRAINTS LIKE L <= AX <= C AND h_1 <= x <= h_2. If you only
    have constraints like L <= AX <= C (our current setting), the projection can be analytically computed 
    and will be much faster to use project_feasible_analytical
    """
    n, T = game_dict["n"], game_dict["T"]
    reserve = game_dict["reserve"]
    assert reserve is not None
    Vs = game_dict["Vs"]

    # quadratic objective
    I = csc_matrix(np.eye(n*T))
    q = -1*np.array(z)

    # leq sum constraints
    I_n = np.eye(n)
    A = np.kron(I_n, np.ones(T))
    A = csc_matrix(A)
    c = np.array(Vs)
    l = -np.inf * np.ones(n)

    prob = osqp.OSQP()
    prob.setup(P=I, q=q, A=A, l=l, u=c, verbose=False)
    res = prob.solve()
    return res.x


def extra_gradient_equilibrium(game_dict, eta=None, eps=0.0001):
    """ Express the equilibrium solution as a joint variational inequality and use the projected extra gradient
    algorithm with step size eta to solve this. At every step, we do a projected look ahead, and the update the
    current value based on the gradient direction from the projected lookahead.
    
    For linear convergence, we need the step size eta such that
    \eta <= 1/L, where L = nTa+a+b(n+1).
    Note this is different than one in paper - that is a more conservative value used for easier convergence proof.
    """
    n, T, Vs = game_dict["n"], game_dict["T"], game_dict["Vs"]
    alpha, beta = game_dict["alpha"], game_dict["beta"]
    vwap_volume = game_dict["vwap_volume"] if "vwap_volume" in game_dict else [1 for i in range(T)]
    fixed_agents = game_dict["vwap_players"] if "vwap_players" in game_dict else []

    M, b = get_linear_operator(game_dict)
    # Initial guess will be VWAP
    vwap_weight = [val/sum(vwap_volume) for val in vwap_volume]
    initial_guess = np.concatenate([[Vs[i] * vwap_weight[t] for t in range(T)] for i in range(n)])
    L = (n*T + 1)*alpha + (n+1)*beta
    if eta is None:
        eta = 0.98/L
    else:
        assert eta <= 1/L
        
    # The fixed agents will play their Vwap strategy. So do not update them ...
    prev_x, curr_x = np.zeros(n*T), initial_guess
    while np.linalg.norm(prev_x-curr_x) >= eps:
        prev_x = curr_x.copy()
        f_prev = np.matmul(M, prev_x) + b
        lookahead_x = prev_x - eta*f_prev
        lookahead_x = project_feasible_analytical(game_dict, lookahead_x)
        f_lookahead = np.matmul(M, lookahead_x) + b

        # Only update the agents that are not fixed
        if len(fixed_agents) == 0:
            curr_x = prev_x - eta*f_lookahead
        else:
            for i in range(n):
                if i not in fixed_agents:
                    curr_x[i*T:(i+1)*T] = prev_x[i*T:(i+1)*T] - eta*f_lookahead[i*T:(i+1)*T]
        curr_x = project_feasible_analytical(game_dict, curr_x)
    
    equi_demand = curr_x.reshape(n, T)
    return equi_demand
     
     

def extra_gradient_equilibrium_bayesian(game_dict, eta=None, eps=0.0001):
    """ Express the equilibrium solution as a joint variational inequality and use the projected extra gradient
    algorithm with step size eta to solve this. At every step, we do a projected look ahead, and the update the
    current value based on the gradient direction from the projected lookahead.
    
    For linear convergence, we need the step size eta such that
    \eta <= 1/L, where L = nTa+a+b(n+1).
    Note this is different than one in paper - that is a more conservative value used for easier convergence proof.
    """
    n, k, T, Vs = game_dict["n"], game_dict["k"], game_dict["T"], game_dict["Vs"]
    alphas, betas = game_dict["alphas"], game_dict["betas"]
    M, b = get_linear_operator_bayesian(game_dict)

    initial_guess = np.concatenate([np.concatenate([[Vs[(i,l)]/T for t in range(T)] for l in range(k)]) for i in range(n)])
    L = (n*T + 1)*np.max(alphas) + (n+1)*np.max(betas)
    if eta is None:
        eta = 0.98/L
    else:
        assert eta <= 1/L

    prev_x, curr_x = np.zeros(n*k*T), initial_guess
    while np.linalg.norm(prev_x-curr_x) >= eps:
        prev_x = curr_x.copy()
        f_prev = np.matmul(M, prev_x) + b
        lookahead_x = prev_x - eta*f_prev
        lookahead_x = project_feasible_analytical_bayesian(game_dict, lookahead_x)
        
        f_lookahead = np.matmul(M, lookahead_x) + b
        curr_x = prev_x - eta*f_lookahead
        curr_x = project_feasible_analytical_bayesian(game_dict, curr_x)
    
    equi_demand = curr_x.reshape(n, k, T)
    return equi_demand


def test_complete_info():
    n, T, alpha, beta = 2, 5, 1, 1
    Vs = [10, 30]
    reserve = [2000 for i in range(n)]
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
    #proj = project_feasible(game_dict, [10, 10, 10, 10, 10, 10, 10, 10])
    start = time.time()
    out = extra_gradient_equilibrium(game_dict)
    end = time.time()
    print(f"Took: {end - start} seconds")
    print(out)


def test_bayesian_1():
    # We have n=2 and k=3. Uniform type distribution
    n, T, k = 2, 5, 2
    bayesian_game_dict = {
        "n" : n,
        "T" : T,
        "k" : k,
        "p_0" : 2.0,
        "supply" : [0 for i in range(T)]
    }
    Vs = np.array([
        [10, 10],
        [30, 30]
    ])
    reserves = np.array([
        [2000, 2000],
        [2000, 2000]
    ])
    bayesian_game_dict["Vs"] = Vs
    bayesian_game_dict["reserves"] = reserves
    
    # key is agent1 type, agent2 type
    alphas, betas, type_dist = np.ones((k,k)), np.ones((k,k)), (1/k**2)*np.ones((k,k))
    bayesian_game_dict["alphas"] = alphas
    bayesian_game_dict["betas"] = betas
    bayesian_game_dict["type_dist"] = type_dist
    out = extra_gradient_equilibrium_bayesian(bayesian_game_dict, eta=None, eps=0.0001)
    print(f"Bayesian out: \n {out}")


def test_bayesian_2():
    # We have n=2 and k=3. Uniform type distribution
    n, k, T = 2, 3, 4
    bayesian_game_dict = {
        "n" : n,
        "T" : T,
        "k" : k,
        "p_0" : 2,
        "supply" : [0 for i in range(T)]
    }

    # Vs and reserves are an n (agent) x k (type) matrix. 
    Vs = np.array([
        [10, 15, 20],
        [20, 25, 30]
    ])
    reserves = Vs/3
    bayesian_game_dict["Vs"] = Vs
    bayesian_game_dict["reserves"] = reserves

    # key is agent1 type, agent2 type
    # All that really matters is the expected value of alpha, beta conditioned on the type. Which is what this is
    alphas, betas, type_dist = np.zeros((k,k)), np.zeros((k,k)), np.zeros((k,k))
    for l0 in range(k):
        for l1 in range(k):
            key = (l0, l1)
            beta = 0.5*(bayesian_game_dict["Vs"][(0,l0)] + bayesian_game_dict["Vs"][(1,l1)])/10
            alpha = 0.1
            alphas[l0, l1] = alpha
            betas[l0, l1] = beta
            type_dist[l0, l1] = 1/k**2

    bayesian_game_dict["alphas"] = alphas
    bayesian_game_dict["betas"] = betas
    bayesian_game_dict["type_dist"] = type_dist
    out = extra_gradient_equilibrium_bayesian(bayesian_game_dict, eta=None, eps=0.0001)
    print(f"Bayesian out: \n {out}")


def test_bayesian_3():
    # We have n=2 and k=3. Uniform type distribution
    n, k, T = 2, 3, 4
    bayesian_game_dict = {
        "n" : n,
        "T" : T,
        "k" : k,
        "p_0" : 0,
        "supply" : [0 for i in range(T)]
    }

    # Vs and reserves are an n (agent) x k (type) matrix. 
    Vs = np.array([
        [10, 15, 20],
        [20, 25, 30]
    ])
    reserves = np.array([
        [1000, 1000, 1000],
        [1000, 1000, 1000]
    ]) 
    bayesian_game_dict["Vs"] = Vs
    bayesian_game_dict["reserves"] = reserves

    # key is agent1 type, agent2 type
    # All that really matters is the expected value of alpha, beta conditioned on the type. Which is what this is
    alphas, betas, type_dist = np.zeros((k,k)), np.zeros((k,k)), np.zeros((k,k))
    for l0 in range(k):
        for l1 in range(k):
            key = (l0, l1)
            beta = 1
            alpha = 1
            alphas[l0, l1] = alpha
            betas[l0, l1] = beta
            type_dist[l0, l1] = 1/k**2

    bayesian_game_dict["alphas"] = alphas
    bayesian_game_dict["betas"] = betas
    bayesian_game_dict["type_dist"] = type_dist
    out = extra_gradient_equilibrium_bayesian(bayesian_game_dict, eta=None, eps=0.0001)
    print(f"Bayesian out: \n {out}")
    for i in range(n):
        for l in range(k):
            total = np.sum(out[i][l])
            print(f"Total bought by agent {i} of type {l} is: {total}")
    return out


if __name__ == "__main__":
    test_complete_info()
    test_bayesian_1()
    #test_bayesian_3()