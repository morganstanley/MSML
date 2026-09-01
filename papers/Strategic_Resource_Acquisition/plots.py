import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from main import find_equilibrium_br, get_cost
from algorithms import extra_gradient_equilibrium, extra_gradient_equilibrium_bayesian
from market_data_experiments import get_supply_vector
from cost_models import get_price_vector

import imageio.v2 as imageio
import os, re, glob
from tqdm import tqdm


# Enable LaTeX text rendering globally
plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath\bfseries' # or other packages that support bold

# Set the font family (e.g., to serif fonts often used with LaTeX)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or other serif fonts

plt.rcParams.update({'font.size': 13}) # Default font size for most text

# # Specific font sizes for different text elements
# plt.rcParams['axes.titlesize'] = 14     # Font size of the axes title
# plt.rcParams['axes.labelsize'] = 12     # Font size of the x and y labels
# plt.rcParams['xtick.labelsize'] = 10    # Font size of the x-axis tick labels
# plt.rcParams['ytick.labelsize'] = 10    # Font size of the y-axis tick labels
# plt.rcParams['legend.fontsize'] = 10    # Font size of the legend
# plt.rcParams['figure.titlesize'] = 16   # Font size of the figure title


def plot_price(ax, game_dict, demand_matrix, supply, plot_y_label=True):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    
    # Get price vector
    price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix)
    time_steps = np.arange(T)

    # Color palette for players
    colors = ['red', 'pink']
    
    price_delta = price_vector
    perm_price_delta = perm_price_vector
    ax.plot(time_steps, price_delta, linewidth=2, color=colors[0], label=r'\textbf{Execution Price: $p_t$}')
    ax.plot(time_steps, perm_price_delta, linewidth=2, color=colors[1], label=r'\textbf{Perm Impact only Price: $p_t^w$}')
    ax.set_xlabel(r'\textbf{Time (minutes)}')
    alpha_mantissa = alpha / (10 ** np.floor(np.log10(np.abs(alpha))))
    alpha_exponent = int(np.floor(np.log10(np.abs(alpha))))  
    beta = beta/T
    beta_mantissa = beta / (10 ** np.floor(np.log10(np.abs(beta))))
    beta_exponent = int(np.floor(np.log10(np.abs(beta))))
    alpha_tile = f"$\\alpha={alpha_mantissa:.2f} e^{{{alpha_exponent}}}$"
    beta_title = f"$\\beta={beta_mantissa:.2f} e^{{{beta_exponent}}}$"
    ax.set_title(r"\textbf{Price Evolution (}" + alpha_tile + ", " + beta_title + ")")
    if plot_y_label:
        ax.set_ylabel(r'\textbf{Price}')
    ax.grid(True, alpha=0.25)
    ax.legend()

def plot_demand(ax, game_dict, demand_matrix, supply, plot_type, plot_y_label=True, real_world=True):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    vwap_players = game_dict["vwap_players"] if "vwap_players" in game_dict else []

    # Get price vector
    price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix)
    time_steps = np.arange(T)

    # Color palette for players
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray']
    to_plot_demand = demand_matrix
    to_plot_supply = supply

    if plot_type == "cumulative":
        to_plot_demand = np.cumsum(demand_matrix, axis=1)
        to_plot_supply = np.cumsum(supply) 
   
    for i in range(n):
        if i in vwap_players:
            label = (r'\textbf{Agent} ' + f'{i}' + r' \textbf{(V=}' + rf'{Vs[i]}' + r'\textbf{)}' + r' \textbf{VWAP}')
            linestyle = "-."
        else:
            label = (r'\textbf{Agent} ' + f'{i}' + r' \textbf{(V=}' + rf'{Vs[i]}' + r'\textbf{)}')
            linestyle = "solid"
        ax.plot(time_steps, to_plot_demand[i], linewidth=2,
            color=colors[i % len(colors)],
            label=label,
            linestyle=linestyle
        )
        
    if sum(supply) != 0:
       ax.plot(time_steps, to_plot_supply, linewidth=2, linestyle=":", color=colors[-1], label=r"\textbf{Exogenous Agent}")
    ax.set_xlabel(r'\textbf{Time}')
    if plot_type == "cumulative" and plot_y_label:
        ax.set_ylabel(r'\textbf{Cumulative Position}') 
    elif plot_y_label:
         ax.set_ylabel('Order')
    
    # For real world experiments:
    if real_world:
        alpha_mantissa = alpha / (10 ** np.floor(np.log10(np.abs(alpha))))
        alpha_exponent = int(np.floor(np.log10(np.abs(alpha))))  
        beta = beta/T
        beta_mantissa = beta / (10 ** np.floor(np.log10(np.abs(beta))))
        beta_exponent = int(np.floor(np.log10(np.abs(beta))))
        alpha_title = f"$\\alpha={alpha_mantissa:.2f} e^{{{alpha_exponent}}}$"
        beta_title = f"$\\beta={beta_mantissa:.2f} e^{{{beta_exponent}}}$"
        ax.set_title(r"\textbf{Equilibrium Strategies (}" + alpha_title + ", " + beta_title + ")")
        ax.set_ylim(-150, 2000) # Set y-axis limit
    else:
        ax.set_title(rf'$\alpha={alpha}$, $\beta={beta}$')
        ax.set_ylim(-15, 40)
    ax.legend()
    ax.grid(True, alpha=0.25)  


def plot_demand_bayesian(ax, game_dict, demand_matrix, supply, plot_type, real_world=True, plot_y_label=True):
    n, k, T = game_dict["n"], game_dict["k"], game_dict["T"]
    alphas, betas, p_0 = game_dict["alphas"], game_dict["betas"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserves = game_dict["reserves"]
    
    time_steps = np.arange(T)

    colors = ['blue', 'orange', 'gray']
    to_plot_demand = demand_matrix
    to_plot_supply = supply

    if plot_type == "cumulative":
        to_plot_demand = np.cumsum(demand_matrix, axis=2)
        to_plot_supply = np.cumsum(supply) 
   
    for i in range(n):
        for l in range(k):
            v, r = Vs[i,l], reserves[i,l]
            ax.plot(time_steps, to_plot_demand[i][l], linewidth=2,
                color=colors[i % len(colors)], alpha=(l*0.3 + 0.4),
                label=(r'\textbf{Agent} ' + f'{i}' + rf'; $V_{i+1}(\theta_{i+1})={v}$')
            )
    ax.set_xlabel(r'\textbf{Time (mins)}')
    if plot_type == "cumulative" and plot_y_label:
        ax.set_ylabel(r'\textbf{Cumulative Position}') 
    elif plot_y_label:
         ax.set_ylabel('Order')
    
    if real_world:
        ax.plot(time_steps, to_plot_supply, linewidth=2, linestyle=":", color=colors[-1], label=r"\textbf{Exogenous Agent}")
        alpha = np.min(alphas)
        alpha_mantissa = alpha / (10 ** np.floor(np.log10(np.abs(alpha))))
        alpha_exponent = int(np.floor(np.log10(np.abs(alpha))))  
        alpha_title = f"$\\alpha={alpha_mantissa:.2f} e^{{{alpha_exponent}}}$"
        
        beta_min = np.min(betas)
        beta_max = np.max(betas)
        beta_min_mantissa = beta_min / (10 ** np.floor(np.log10(np.abs(beta_min))))
        beta_min_exponent = int(np.floor(np.log10(np.abs(beta_min))))
        beta_max_mantissa = beta_max / (10 ** np.floor(np.log10(np.abs(beta_max))))
        beta_max_exponent = int(np.floor(np.log10(np.abs(beta_max))))
        beta_title = f"$\\beta \in [{beta_min_mantissa:.2f} e^{{{beta_min_exponent}}}, {beta_max_mantissa:.2f} e^{{{beta_max_exponent}}}]$"
        
        ax.set_title(alpha_title + ", " + beta_title)
        ax.set_ylim(-400, 1000) # Set y-axis limit
        ax.legend(loc='lower right')
    else:
        alpha = np.min(alphas)
        beta_min = np.min(betas)
        beta_max = np.max(betas)
        ax.set_title(rf'$\alpha={alpha}$, $\beta \in [{beta_min}, {beta_max}]$')
        ax.set_ylim(-15, 35) # Set y-axis limit
        ax.legend()
   
    ax.grid(True, alpha=0.25)  


def plot_demand_bayesian_last_iterate(ax, game_dict, demand_matrix_last_iterate, demand_matrix_BNE, supply, plot_type, plot_y_label=True):
    n, k, T = game_dict["n"], game_dict["k"], game_dict["T"]
    alphas, betas, p_0 = game_dict["alphas"], game_dict["betas"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserves = game_dict["reserves"]
    
    time_steps = np.arange(T)

    colors = ['blue', 'orange']
    to_plot_demand_last_iterate = demand_matrix_last_iterate
    to_plot_demand_BNE = demand_matrix_BNE
    to_plot_supply = supply

    if plot_type == "cumulative":
        to_plot_demand_last_iterate = np.cumsum(demand_matrix_last_iterate, axis=2)
        to_plot_demand_BNE = np.cumsum(demand_matrix_BNE, axis=2)
        to_plot_supply = np.cumsum(supply) 
   
    for i in range(n):
        for l in range(k):
            v, r = Vs[i,l], reserves[i,l]
            ax.plot(time_steps, to_plot_demand_last_iterate[i][l], linewidth=2,
                color=colors[i % len(colors)], alpha=(l*0.3 + 0.4),
                label=(r'\textbf{Agent} ' + f'{i+1}' + rf' $\theta_{i}={v}$ (Last It.)')
            )
            ax.plot(time_steps, to_plot_demand_BNE[i][l], '--', linewidth=2,
                color=colors[i % len(colors)], alpha=(l*0.3 + 0.4),
                label=(r'\textbf{Agent} ' + f'{i+1}' + rf' $\theta_{i}={v}$ (BNE)')
            )
    ax.set_xlabel(r'\textbf{Time}', fontsize=18)
    if plot_type == "cumulative" and plot_y_label:
        ax.set_ylabel(r'\textbf{Cumulative Position}', fontsize=18) 
    elif plot_y_label:
         ax.set_ylabel('Order', fontsize=18)
    
    alpha = np.min(alphas)
    beta_min = np.min(betas)
    beta_max = np.max(betas)
    # ax.set_title(rf'$\alpha={alpha}$, $\beta \in [{beta_min}, {beta_max}]$')
    ax.set_ylim(-15, 35) # Set y-axis limit
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.legend()
    ax.grid(True, alpha=0.25) 


def plot_equilibrium_strategies(game_dict, demand_matrix, supply, exp_number=None, beta_ab=None, time_ab=None):
    """
    Plot the equilibrium strategies showing:
    1. Individual player strategies over time
    2. Cumulative positions for each player
    3. Supply over time
    4. Price evolution over time (with reserve prices)
    """
    discretization, cont_time_interval = game_dict["discretization"], game_dict["cont_time_interval"]
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    
    # Get price vector
    price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    time_steps = np.arange(T)
    
    # Color palette for players
    colors = ['blue', 'orange', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    ticks = np.arange(0, (cont_time_interval+1) * discretization, discretization)
    
    # 1. Individual player strategies (orders per time step)
    ax1 = axes[0, 0]
    plot_demand(ax1, game_dict, demand_matrix, supply, "order")
    
    # 2. Cumulative positions (with alternating plotting order)
    ax2 = axes[0, 1]
    plot_demand(ax2, game_dict, demand_matrix, supply, "cumulative")

    # Plot costs
    ax3 = axes[1, 0]
    # Elementwise multiply each row of demand_matrix by price_vector
    costs_per_step = demand_matrix * price_vector  # shape (n, T)

    # Cumulative sum along time axis (axis=1)
    cost_matrix = np.cumsum(costs_per_step, axis=1)  # shape (n, T)
    for i in range(n):
        ax3.plot(time_steps, cost_matrix[i], linewidth=2,
            color=colors[i % len(colors)],
            label=(f'Trader {i} (V={Vs[i]}, R={reserve[i]})')
        )
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Cumulative cost upto t')
    ax3.set_title('Cumulative Costs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels((ticks // discretization).astype(int))

    # 4. Price evolution (with reserve prices)
    ax4 = axes[1, 1]
    plot_price(ax4, game_dict, demand_matrix, supply)
    
    # Add overall title
    fig.suptitle(f'Equilibrium Analysis: n:{n}, Discretization steps: {discretization}, Cont time: {cont_time_interval}, T: {discretization}*{cont_time_interval}, α={alpha}, β={beta/discretization:.3f}*{discretization})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if exp_number:
        if time_ab:
            plt.savefig(f"figures/exp{exp_number}_T/exp{exp_number}_T_{cont_time_interval}.png")
        elif beta_ab:
            beta_int = int(beta/discretization * 1000)
            plt.savefig(f"figures/exp{exp_number}_beta/exp{exp_number}_beta_{beta_int}.png")
        plt.close()
    else:
        plt.show()


def extract_number(filename):
    # This regex finds the last group of digits in the filename
    match = re.search(r'(\d+)(?=\.png$)', filename)
    return int(match.group(1)) if match else -1


def bayesian_experiment():
    n, k, T = 2, 3, 100
    supply = [0 for i in range(T)]
    bayesian_game_dict = {
        "n" : n,
        "T" : T,
        "k" : k,
        "p_0" : 2,
        "supply" : supply
    }

    # Vs and reserves are an n (agent) x k (type) matrix. 
    Vs = np.array([
        [10, 15, 20],
        [20, 25, 30]
    ])
    reserves = np.array([
        [3, 5, 7],
        [6, 8, 10]
    ]) 
    bayesian_game_dict["Vs"] = Vs
    bayesian_game_dict["reserves"] = reserves

    # key is agent1 type, agent2 type
    # All that really matters is the expected value of alpha, beta conditioned on the type. Which is what this is
    alphas, betas, type_dist = np.zeros((k,k)), np.zeros((k,k)), np.zeros((k,k))
    for l0 in range(k):
        for l1 in range(k):
            key = (l0, l1)
            beta = 0.5*(bayesian_game_dict["Vs"][(0,l0)] + bayesian_game_dict["Vs"][(1,l1)])/200
            alpha = 0.1
            alphas[l0, l1] = alpha
            betas[l0, l1] = beta
            type_dist[l0, l1] = 1/k**2

    bayesian_game_dict["alphas"] = alphas
    bayesian_game_dict["type_dist"] = type_dist

    beta_multiplier = [1, 10, 100]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, mult in enumerate(beta_multiplier):
        bayesian_game_dict["betas"] = mult*betas
        demand_matrix = extra_gradient_equilibrium_bayesian(bayesian_game_dict)
        plot_demand_bayesian(axes[i], bayesian_game_dict, demand_matrix, supply, "cumulative", plot_y_label=(True if i==0 else False), real_world=False)
    plt.tight_layout()
    plt.show()


def complete_information_experiment():
    # If you're doing cont time:
    # choose discretization = d
    # choose cont_time_interval = c
    # choose T = d*c
    # choose beta*discretization
    n, alpha = 5, 0.1
    T = 100
    Vs = [10, 15, 20, 25, 30]
    reserve = [4, 5, 6, 7, 8]
    beta_range = [0.1, 1, 10]
    supply = np.random.randn(T)*0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, beta in enumerate(beta_range):
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
        demand_matrix = extra_gradient_equilibrium(game_dict)
        plot_demand(axes[i], game_dict, demand_matrix, supply, "cumulative", plot_y_label=(True if i==0 else False), real_world=False)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, beta in enumerate(beta_range):
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
        demand_matrix = extra_gradient_equilibrium(game_dict)
        plot_price(axes[i], game_dict, demand_matrix, supply, plot_y_label=(True if i==0 else False))
    plt.tight_layout()
    plt.show() 


def vwap_experiment_single():
    n, alpha = 5, 0.1
    T = 100
    Vs = [10, 15, 20, 25, 30]
    reserve = [500, 500, 500, 500, 500]
    supply = np.random.randn(T)*0.5

    # First plot the VWAP strategies
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    beta_range = [0.1, 1, 10]
    for i, beta in enumerate(beta_range):
        game_dict = {
            "n" :   n,
            "T" :   T,
            "p_0" : 2.0,
            "Vs" : Vs,
            "alpha" : alpha,
            "beta" : beta,
            "supply" : supply,
            "reserve" : reserve,
            "exp" : 1,
            "vwap_players" : [2,3]
        }
        demand_matrix = extra_gradient_equilibrium(game_dict)
        plot_demand(axes[i], game_dict, demand_matrix, supply, "cumulative", plot_y_label=(True if i==0 else False), real_world=False)
    plt.tight_layout()
    plt.show()

    # Plot welfare comparison
    beta_range = [0.1, 0.3, 0.6, 1, 1.5, 2.5, 4, 6, 8, 10]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    vwap_cost_array = np.zeros((len(beta_range), n))
    equi_cost_array = np.zeros((len(beta_range), n))

    for i, beta in enumerate(beta_range):
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

        # all are playing strategic
        demand_matrix = extra_gradient_equilibrium(game_dict)
        alg_equi_price, _ = get_price_vector(game_dict, demand_matrix)
        strat_costs = [np.dot(alg_equi_price, demand_matrix[k]) for k in range(n)]
        equi_cost_array[i] = strat_costs

        game_dict["vwap_players"] = [2,3]
        demand_matrix = extra_gradient_equilibrium(game_dict)
        alg_equi_price, _ = get_price_vector(game_dict, demand_matrix)
        vwap_costs = [np.dot(alg_equi_price, demand_matrix[k]) for k in range(n)]
        vwap_cost_array[i] = vwap_costs

    # Plot cost ratios for agents who switched from strategic to VWAP (agents 2 and 3)
    x_vals = beta_range
    ratio_2 = vwap_cost_array[:,2] / equi_cost_array[:,2]
    ratio_3 = vwap_cost_array[:,3] / equi_cost_array[:,3]
    axes[0].plot(x_vals, ratio_2, linewidth=2, marker='o', color='purple', label=r'\textbf{Agent 2 (switched to VWAP)}', linestyle="-.")
    axes[0].plot(x_vals, ratio_3, linewidth=2, marker='s', color='green', label=r'\textbf{Agent 3 (switched to VWAP)}', linestyle="-.")
    axes[0].axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label=r'\textbf{Equal Cost (ratio=1)}')
    axes[0].set_xlabel(r'$\beta$ $(\alpha=0.1)$')
    axes[0].set_ylabel(r'\textbf{VWAP Cost / All Strategic Cost}')
    axes[0].set_title(r'\textbf{Cost Ratio: Agents Who Switched}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # Plot cost ratios for agents who stayed strategic (agents 0, 1, and 4)
    ratio_0 = vwap_cost_array[:,0] / equi_cost_array[:,0]
    ratio_1 = vwap_cost_array[:,1] / equi_cost_array[:,1]
    ratio_4 = vwap_cost_array[:,4] / equi_cost_array[:,4]
    axes[1].plot(x_vals, ratio_0, linewidth=2, marker='o', color='blue', label=r'\textbf{Agent 0 (stayed strategic)}')
    axes[1].plot(x_vals, ratio_1, linewidth=2, marker='s', color='orange', label=r'\textbf{Agent 1 (stayed strategic)}')
    axes[1].plot(x_vals, ratio_4, linewidth=2, marker='^', color='brown', label=r'\textbf{Agent 4 (stayed strategic)}')
    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label=r'\textbf{Equal Cost (ratio=1)}')
    axes[1].set_xlabel(r'$\beta$ $(\alpha=0.1)$')
    axes[1].set_ylabel(r'\textbf{others VWAP / All Strategic Cost}')
    axes[1].set_title(r'\textbf{Cost Ratio: Agents Who Stayed Strategic}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def real_market_experiment_complete():
    # If you're doing cont time:
    # choose discretization = d
    # choose cont_time_interval = c
    # choose T = d*c
    # choose beta*discretization
    c = 1                                       # break up 1 hour into T discrete steps
    num_ticks = 6000                            # corresponds to an hour of ticks
    T = 60                                      # Want decisions make every minute
    num_ticks_in_interval = num_ticks/T
    
    supply, total_volume = get_supply_vector()
    to_edit = [supply, total_volume]
    for i, arr in enumerate(to_edit):
        arr = arr[:num_ticks]
        arr = arr.groupby(arr.index // num_ticks_in_interval).sum().repeat(1)
        arr = arr.to_numpy()
        to_edit[i] = arr
    supply, total_volume = to_edit[0], to_edit[1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    n, alpha, beta = 5, 4.65e-7, 3.25e-6*T
    Vs = [500, 600, 700, 800, 900]
    T = len(supply)
    reserve = [1.400, 1.405, 1.410, 1.415, 1.420]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 1.395,
        "Vs" : Vs,
        "alpha" : alpha,
        "beta" : beta,
        "supply" : supply,
        "reserve" : reserve,
        "exp" : 1,
        "vwap_volume" : total_volume
    }
    demand_matrix = extra_gradient_equilibrium(game_dict)
    plot_demand(axes[0], game_dict, demand_matrix, supply, "cumulative", plot_y_label=True)
    plot_price(axes[1], game_dict, demand_matrix, supply, plot_y_label=True)
    plt.tight_layout()
    plt.show() 

def real_market_experiment_bayesian():
    # If you're doing cont time:
    # choose discretization = d
    # choose cont_time_interval = c
    # choose T = d*c
    # choose beta*discretization
    c = 1                                       # break up 1 hour into T discrete steps
    num_ticks = 6000                            # corresponds to an hour of ticks
    T = 60                                      # Want decisions make every minute
    num_ticks_in_interval = num_ticks/T
    
    supply, total_volume = get_supply_vector()
    to_edit = [supply, total_volume]
    for i, arr in enumerate(to_edit):
        arr = arr[:num_ticks]
        arr = arr.groupby(arr.index // num_ticks_in_interval).sum().repeat(1)
        arr = arr.to_numpy()
        to_edit[i] = arr
    supply, total_volume = to_edit[0]*(0.5), to_edit[1]

    n, k = 2, 3
    bayesian_game_dict = {
        "n" : n,
        "T" : T,
        "k" : k,
        "p_0" : 1.395,
        "supply" : supply
    }

    # Vs and reserves are an n (agent) x k (type) matrix. 
    Vs = np.array([
        [500, 600, 700],
        [800, 850, 900]
    ])
    reserves = np.array([
        [1.400, 1.405, 1.410],
        [1.415, 1.420, 1.425]
    ]) 
    bayesian_game_dict["Vs"] = Vs
    bayesian_game_dict["reserves"] = reserves

    # key is agent1 type, agent2 type
    # All that really matters is the expected value of alpha, beta conditioned on the type. Which is what this is
    alphas, betas, type_dist = np.zeros((k,k)), np.zeros((k,k)), np.zeros((k,k))
    
    alphas = np.ones((k,k))*4.65e-7
    betas = np.ones((k,k))*3.25e-6 + np.random.uniform(low=-1e-7, high=1e-7, size=(k,k))
    type_dist = np.ones((k,k))*(1/k**2)

    bayesian_game_dict["alphas"] = alphas
    bayesian_game_dict["type_dist"] = type_dist

    beta_multiplier = [1, 10]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for i, mult in enumerate(beta_multiplier):
        bayesian_game_dict["betas"] = mult*betas
        demand_matrix = extra_gradient_equilibrium_bayesian(bayesian_game_dict)
        print(demand_matrix)
        plot_demand_bayesian(axes[i], bayesian_game_dict, demand_matrix, supply, "cumulative", plot_y_label=(True if i==0 else False))
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    #vwap_experiment_single()
    #real_market_experiment_complete()
    #complete_information_experiment()
    #bayesian_experiment()
    real_market_experiment_bayesian()

    # print("Completed generating plots") 
    # image_files = sorted(glob.glob(f'figures/exp{exp_number}_{ablation}/exp{exp_number}_{ablation}_*.png'), key=extract_number)
    # with imageio.get_writer(f'ablation_exp{exp_number}_{ablation}.gif', mode='I', fps=1.8) as writer:
    #     for filename in image_files:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

