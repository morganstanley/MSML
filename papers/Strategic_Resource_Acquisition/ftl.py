import numpy as np
from main import get_buyer_best_response
from cost_models import get_price_vector
from tqdm import tqdm

def get_costs(game_dict, demand_matrix):
    """Compute total cost for each player given their strategies.
    
    Args:
        demand_matrix (array): Demand matrix
        game_dict (dict): Dictionary containing game parameters
    """
    reserve = game_dict["reserve"]
    pts, _ = get_price_vector(game_dict, demand_matrix)
    costs = []
    
    for i in range(game_dict["n"]):
        cost_i = np.dot(pts, demand_matrix[i])
        if reserve is not None:
            cost_i -= reserve[i] * np.sum(demand_matrix[i])
        costs.append(cost_i)
    
    return costs

def get_regrets(game_dict, actions, costs):
    """Compute cumulative regret for each player."""
    n = game_dict["n"]
    num_rounds = len(actions[0])
    regrets = []

    if num_rounds == 0:
        return [0] * n

    # Get historical average actions for all players
    avg_actions = [np.mean(actions_i, axis=0) for actions_i in actions]
    avg_demand_matrix = np.array(avg_actions)
    
    for i in range(n):
        # 1. Find best fixed action in hindsight for player i.
        best_action_i = get_buyer_best_response(game_dict, avg_demand_matrix, i)

        demand_matrix = avg_demand_matrix.copy()
        demand_matrix[i] = best_action_i
        best_action_cost = num_rounds * get_costs(game_dict, np.array(demand_matrix))[i]

        actual_cost = np.sum(costs[i])
        regrets.append(actual_cost - best_action_cost)
    
    return regrets

class FTL:
    """
    Represents a single FTL algorithm for a player. It stores the history
    of all players' actions observed when this algorithm was active.
    """
    def __init__(self, player_id, n, T, V):
        self.player_id = player_id
        self.n = n
        self.T = T
        self.V = V
        # Stores this algorithm's view of the history of all players' actions
        self.action_history = [[] for _ in range(n)]

    def get_initial_action(self):
        """Returns a random set of actions for the first round in a context."""
        # random_actions = np.random.rand(self.T)
        # if np.sum(random_actions) > 0:
        #     return (self.V / np.sum(random_actions)) * random_actions
        
        return np.ones(self.T) * (self.V / self.T)

    def record_actions(self, full_action_profile):
        """Adds the actions of all players for a round to this algorithm's history."""
        for i in range(self.n):
            self.action_history[i].append(full_action_profile[i])

    def get_average_demand_matrix(self):
        """Computes the average demand matrix from this algorithm's observed history."""
        if not self.action_history[0]:
            return np.zeros((self.n, self.T))
        return np.array([np.mean(h, axis=0) for h in self.action_history])


class ContextualFTLDynamics:
    """Manages a Follow-the-Leader simulation across multiple contexts."""
    def __init__(self, contexts, private_info=False):
        self.contexts = contexts
        self.private_info = private_info
        # number of players and time horizon are the same for all contexts
        self.n = self.contexts[0]['n']
        self.T = self.contexts[0]['T']

        # This attribute is needed for private regret calculations.
        # We assume each player's set of possible contexts is the same as the true set.
        self.player_contexts = [contexts for _ in range(self.n)]
        
        if private_info:
            # Each player creates algorithms based on their private observable tuple
            self.player_algs = []
            self.player_context_mapping = []  # Maps context_id to algorithm index for each player
            
            for i in range(self.n):
                player_algs_i = []
                context_mapping_i = {}
                seen_tuples = {}  # Maps tuple to algorithm index
                
                for ctx_id, ctx in enumerate(self.contexts):
                    # Create tuple of what player i can observe
                    if ctx['reserve'] is not None:
                        observable_tuple = (ctx['alpha'], ctx['beta'], tuple(ctx['supply']), ctx['Vs'][i], ctx['reserve'][i])
                    else:
                        observable_tuple = (ctx['alpha'], ctx['beta'], tuple(ctx['supply']), ctx['Vs'][i])
                    
                    if observable_tuple in seen_tuples:
                        # Reuse existing algorithm
                        alg_idx = seen_tuples[observable_tuple]
                        context_mapping_i[ctx_id] = alg_idx
                    else:
                        # Create new algorithm
                        alg_idx = len(player_algs_i)
                        player_algs_i.append(FTL(i, self.n, ctx['T'], ctx['Vs'][i]))
                        seen_tuples[observable_tuple] = alg_idx
                        context_mapping_i[ctx_id] = alg_idx
                
                self.player_algs.append(player_algs_i)
                self.player_context_mapping.append(context_mapping_i)
        else:
            # Original behavior: one algorithm per context
            self.player_algs = [
                [FTL(i, self.n, ctx['T'], ctx['Vs'][i]) for ctx in self.contexts]
                for i in range(self.n)
            ]
            # Simple mapping: context_id maps directly to algorithm index
            self.player_context_mapping = [
                {ctx_id: ctx_id for ctx_id in range(len(self.contexts))}
                for _ in range(self.n)
            ]

        # Store simulation results
        self.overall_actions = [[] for _ in range(self.n)]
        self.overall_costs = [[] for _ in range(self.n)]
        self.per_context_regrets = [[[] for _ in range(len(self.contexts))] for _ in range(self.n)]
        self.contextual_regrets = [[] for _ in range(self.n)]
        self.private_per_context_regrets = [[[] for _ in range(len(self.contexts))] for _ in range(self.n)]
        self.private_contextual_regrets = [[] for _ in range(self.n)]

    def run(self, context_sequence, num_iter):
        """Executes the contextual FTL simulation."""
        if len(context_sequence) < num_iter:
            raise ValueError("Length of context_sequence must be at least num_iter.")

        for r in tqdm(range(num_iter), desc="Contextual FTL"):
            context_id = context_sequence[r]
            
            # --- Action Selection ---
            current_round_actions = []
            for i in range(self.n):
                alg_idx = self.player_context_mapping[i][context_id]
                current_alg = self.player_algs[i][alg_idx]
                
                if not current_alg.action_history[0]:
                    action = current_alg.get_initial_action()
                else:
                    avg_demand_matrix = current_alg.get_average_demand_matrix()
                    action = get_buyer_best_response(self.contexts[context_id], avg_demand_matrix, i)
                current_round_actions.append(action)

            # --- Cost Calculation & History Update ---
            costs_r = get_costs(self.contexts[context_id], np.array(current_round_actions))
            for i in range(self.n):
                alg_idx = self.player_context_mapping[i][context_id]
                self.player_algs[i][alg_idx].record_actions(current_round_actions)
                self.overall_actions[i].append(current_round_actions[i])
                self.overall_costs[i].append(costs_r[i])

            # --- Regret Calculations ---
            for i in range(self.n):
                # Objective Regret (per true context)
                true_game_dict = self.contexts[context_id]
                obj_indices = [k for k, ctx_id in enumerate(context_sequence[:r+1]) if ctx_id == context_id]
                sub_regret = _compute_regret_on_subsequence(i, obj_indices, true_game_dict, self.overall_actions, self.overall_costs)
                self.per_context_regrets[i][context_id].append(sub_regret if sub_regret is not None else 0)
                
                total_contextual_regret = sum(self.per_context_regrets[i][c_id][-1] for c_id in range(len(self.contexts)) if self.per_context_regrets[i][c_id])
                self.contextual_regrets[i].append(total_contextual_regret)

                # Private Regret (per observable context / algorithm)
                private_game_dict = self.player_contexts[i][context_id]
                target_alg_idx = self.player_context_mapping[i][context_id]
                priv_indices = [k for k in range(r + 1) if self.player_context_mapping[i][context_sequence[k]] == target_alg_idx]
                priv_sub_regret = _compute_regret_on_subsequence(i, priv_indices, private_game_dict, self.overall_actions, self.overall_costs)
                # self.private_per_context_regrets[i][context_id].append(priv_sub_regret if priv_sub_regret is not None else 0)
                self.private_per_context_regrets[i][target_alg_idx].append(priv_sub_regret if priv_sub_regret is not None else 0)
                
                # Correctly sum unique algorithm regrets for total private contextual regret
                total_private_regret = 0
                summed_algs = set()
                for ctx_id_inner in range(len(self.contexts)):
                    alg_idx_inner = self.player_context_mapping[i][ctx_id_inner]
                    if alg_idx_inner not in summed_algs and self.private_per_context_regrets[i][ctx_id_inner]:
                        total_private_regret += self.private_per_context_regrets[i][ctx_id_inner][-1]
                        summed_algs.add(alg_idx_inner)
                self.private_contextual_regrets[i].append(total_private_regret)

        return self.overall_actions, self.overall_costs, self.per_context_regrets, self.contextual_regrets, self.private_per_context_regrets, self.private_contextual_regrets

def _compute_regret_on_subsequence(player_id, subsequence_indices, game_dict, overall_actions, overall_costs):
    """
    A general helper to compute regret on a given subsequence of rounds using a
    specific game_dict (either true or a player's private belief).
    """
    if not subsequence_indices:
        return None

    sub_actions = [[overall_actions[p][r] for r in subsequence_indices] for p in range(len(overall_actions))]
    sub_costs_player = [overall_costs[player_id][r] for r in subsequence_indices]
    
    formatted_sub_costs = [[] for _ in range(len(overall_actions))]
    formatted_sub_costs[player_id] = sub_costs_player
    
    return get_regrets(game_dict, sub_actions, formatted_sub_costs)[player_id]

def get_last_iterate_strategies(dynamics, overall_actions, context_sequence):
    """
    Returns the last-iterate strategies for each player for each of their observable contexts.

    Args:
        dynamics (ContextualFTLDynamics): The dynamics object after running the simulation.
        overall_actions (list): The overall_actions output from the run() function.
        context_sequence (list): The sequence of contexts used in the simulation.
    """
    n = dynamics.n
    T = dynamics.T
    num_iter = len(context_sequence)

    # Find the last round each algorithm was used for each player
    last_rounds = {}  # {(player_id, alg_idx): round}
    for r in range(num_iter):
        context_id = context_sequence[r]
        for i in range(n):
            alg_idx = dynamics.player_context_mapping[i][context_id]
            last_rounds[(i, alg_idx)] = r
            
    last_iterate_strategies = [[] for _ in range(n)]
    for i in range(n):
        num_player_algs = len(dynamics.player_algs[i])
        for alg_idx in range(num_player_algs):
            if (i, alg_idx) in last_rounds:
                last_round = last_rounds[(i, alg_idx)]
                strategy = overall_actions[i][last_round]
                last_iterate_strategies[i].append(strategy)
    
    return last_iterate_strategies

