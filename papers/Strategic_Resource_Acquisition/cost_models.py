import numpy as np
import matplotlib.pyplot as plt

def chriss_model(demand_matrix, supply_vector, p_init, alpha=1, use_supply=False, use_initial_price=False):
    # The original Chriss model does NOT use any supply in the formulation
    # The conditional usage of supply here was for me to personally verify some aspect of this model.
    # For standard usage of the Chriss model, set use_supply=False and use_initial_price=False
    
    n, T = demand_matrix.shape
    temp_price_vector = np.zeros(T)
    permanent_price_vector = np.zeros(T)
    total_price_vector = np.zeros(T)

    for t in range(T):
        temp_price_vector[t] = np.sum(demand_matrix[:, t])
        if use_supply:
            temp_price_vector[t] -= supply_vector[t]
        
        permanent_price_vector[t] = sum([np.sum(demand_matrix[:, i]) for i in range(t)]) 
        if use_supply:
            permanent_price_vector[t] -= np.sum(supply_vector[:t])
       
    total_price_vector = temp_price_vector + alpha*permanent_price_vector 
    if use_initial_price:
        total_price_vector += np.array([p_init for i in range(T)])
    return temp_price_vector, permanent_price_vector, total_price_vector


def get_price_vector(game_instance, demand_matrix, i=None, demand_i=None):
    """ Computing price is tricky and we need to do it multiple times for various objective/optimization.
        They should all use this function and any changes should be reflected here.

        If demand_i is set, we are essentially doing a best-response for i (hence it's handled seperately)
    """
    n, T = game_instance["n"], game_instance["T"]
    supply_vector = game_instance["supply"]
    p_init = game_instance["p_0"]
    alpha = game_instance["alpha"]
    beta = game_instance["beta"]
    price_vector = np.zeros(T)
    perm_price_vector = np.zeros(T)
    
    # This is equivalent - kept here for reference
    # for t in range(0, T):
    #     if t == 0:
    #         price_vector_walruss[t] = p_init + alpha*(np.sum(demand_matrix[:,t]) - supply_vector[t])
    #     else:
    #         price_vector_walruss[t] = price_vector_walruss[t-1] + alpha*(np.sum(demand_matrix[:,t]) - supply_vector[t])
    #     price_vector[t] = price_vector_walruss[t] + beta*(np.sum(demand_matrix[:, t]) - supply_vector[t])

    exp = game_instance["exp"]
    
    for t in range(T):
        pt = p_init
        for l in range(t+1):
            if demand_i is not None:
                pt += alpha*(np.sum(demand_matrix[:, l]) - demand_matrix[i,l] + demand_i[l] - supply_vector[l])
            else:
                pt += alpha*(np.sum(demand_matrix[:, l]) - supply_vector[l])

        if demand_i is not None:
            net_demand = np.sum(demand_matrix[:, t]) - demand_matrix[i, t] + demand_i[t] - supply_vector[t]
        else:
            net_demand = np.sum(demand_matrix[:, t]) - supply_vector[t] 
        
        # Even though the general expression is equivalent to the exp=1 expression when exp=1, I write them seperately
        # since the inclusion of abs leads to SLSQP optimizer complaining.
        perm_price_vector[t] = pt
        if exp == 1:
            pt += beta*net_demand
        else:
            pt += beta*np.sign(net_demand)*(np.abs(net_demand)**exp)
        price_vector[t] = pt

    return price_vector, perm_price_vector


def test_random_data():
    # Verify that the Walrasiaan model can indeed capture the Chriss model 
    # for the correct parametrization
    n, T = 2, 10
    max_val = 101
    demand_matrix = np.random.randint(0, max_val, (n, T))
    #supply_vector = np.random.randint(0, max_val*n, T)
    supply_vector = np.zeros(T)

    print(f"Demand Matrix: {demand_matrix}")
    print(f"Supply Vector: {supply_vector}")
    p_init = 0
    alpha = 1

    _, _, chriss_price = chriss_model(
        demand_matrix, 
        supply_vector, 
        p_init,
        use_supply=True, 
        use_initial_price=True
    )
    walruss_price = walruss_model(
        demand_matrix, 
        supply_vector, 
        p_init, 
        alpha=alpha
    )

    plt.figure(figsize=(10, 6))
    plt.plot(chriss_price, label='Chriss Price')
    plt.plot(walruss_price, label='Walruss Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Chriss vs Walruss Price')
    plt.legend()
    plt.show()


def verify_discrete_equilibrium(actions, game_matrix):
    equilibrium = np.ones((len(actions), len(actions)))
    for row in range(len(game_matrix)):
        for col in range(len(game_matrix[0])):
            row_cost, col_cost = game_matrix[row, col][0], game_matrix[row, col][1]
            # see if there is profitable row deviation
            for deviat_row in range(len(game_matrix)):
                if game_matrix[deviat_row][col][0] < row_cost:
                    equilibrium[row, col] = 0
                    break

            for deviat_col in range(len(game_matrix[0])):
                if game_matrix[row][deviat_col][1] < col_cost:
                    equilibrium[row, col] = 0
                    break
    return equilibrium



def test_discrete_equilibrium():
    def generate_action_pairs(n=4, total=100):
        actions = []
        for _ in range(n):
            a = np.random.randint(0, total + 1)
            b = total - a
            actions.append((a, b))
        return actions
    
    #actions = [(1,99), (20, 80), (80,20), (99,1)]
    #supply_vector = [20, 100]
    for i in range(1000):
        actions = generate_action_pairs()
        supp_0 = np.random.randint(0, 101) 
        supply_vector = [supp_0, 100-supp_0]

        game_matrix = np.array([[(0,0) for i in range(len(actions))] for j in range(len(actions))])
        p0 = 100

        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions):
                demand_matrix = np.array([action1, action2])
                ps = walruss_model(demand_matrix, supply_vector, p_init=p0, alpha=1)
                u1, u2 = np.dot(ps, demand_matrix[0]), np.dot(ps, demand_matrix[1])
                game_matrix[i,j] = (u1, u2)

        equilibrium = verify_discrete_equilibrium(actions, game_matrix)
        indices = np.argwhere(equilibrium == 1)
        print(indices)
        if 1.0 not in equilibrium:
            print("Equilibrium Does Not Exist")

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.set_xticks(range(len(actions)))
    # ax.set_yticks(range(len(actions)))
    # ax.set_xticklabels([str(a) for a in actions])
    # ax.set_yticklabels([str(a) for a in actions])
    # ax.set_xlabel("Player 2 Action")
    # ax.set_ylabel("Player 1 Action")
    # ax.set_title("Game Matrix (Payoff Tuples)")

    # # Draw grid
    # for i in range(len(actions)):
    #     for j in range(len(actions)):
    #         ax.text(j, i, str(equilibrium[i, j]), va='center', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

    # ax.set_xlim(-0.5, len(actions)-0.5)
    # ax.set_ylim(-0.5, len(actions)-0.5)
    # ax.invert_yaxis()
    # plt.grid(True, which='both', color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    # supply_vector = [150, 10]
    # p0 = 100

    # # strat 1
    # ps = walruss_model(demand_matrix_1, supply_vector, p_init=p0, alpha=1)
    # u1, u2 = np.dot(ps, demand_matrix_1[0]), np.dot(ps, demand_matrix_1[1])
    # print(f"Strategy 1: Trader 1 utility: {u1}; Trader 2 utility: {u2}") 

    # # strat 1
    # ps = walruss_model(demand_matrix_2, supply_vector, p_init=p0, alpha=1)
    # u1, u2 = np.dot(ps, demand_matrix_2[0]), np.dot(ps, demand_matrix_2[1])
    # print(f"Strategy 2: Trader 1 utility: {u1}; Trader 2 utility: {u2}") 

    # # strat 1
    # ps = walruss_model(demand_matrix_3, supply_vector, p_init=p0, alpha=1)
    # u1, u2 = np.dot(ps, demand_matrix_3[0]), np.dot(ps, demand_matrix_3[1])
    # print(f"Strategy 3: Trader 1 utility: {u1}; Trader 2 utility: {u2}") 

    # # strat 1
    # ps = walruss_model(demand_matrix_4, supply_vector, p_init=p0, alpha=1)
    # u1, u2 = np.dot(ps, demand_matrix_4[0]), np.dot(ps, demand_matrix_4[1])
    # print(f"Strategy 4: Trader 1 utility: {u1}; Trader 2 utility: {u2}")

    # ps = walruss_model(demand_matrix_5, supply_vector, p_init=p0, alpha=1)
    # u1, u2 = np.dot(ps, demand_matrix_5[0]), np.dot(ps, demand_matrix_5[1])
    # print(f"Strategy 5: Trader 1 utility: {u1}; Trader 2 utility: {u2}")  


if __name__ == "__main__":
    #test_random_data()
    test_discrete_equilibrium()
