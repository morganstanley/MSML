from typing import Union, List
import random
from collections import deque
import csv
import networkx as nx


class OptList:
    def __init__(self):
        self.a, self.d, self.index = [], {}, 0
     
    def __contains__(self, v) -> bool:
        return v in self.d
    
    def __str__(self):
        # Define how you want your object to be 
        return f"({self.a})"
    
    def __iter__(self):
        # The iterator should return itself in this case
        return self

    def __next__(self):
        # Check if there are more elements in the data
        if self.index < len(self.a):
            # Get the next element and increment the index
            result = self.a[self.index]
            self.index += 1
            return result
        else:
            self.index = 0
            # If there are no more elements, raise StopIteration
            raise StopIteration
            
    def __len__(self) -> int:
        return len(self.a)

    def insert(self, v: int) -> bool:
        if v in self.d:
            return False

        self.d[v] = len(self.a)
        self.a.append(v)

        return True
    
    def append(self, v: int) -> bool:
        if v in self.d:
            return False

        self.d[v] = len(self.a)
        self.a.append(v)

        return True
        

    def remove(self, v: int) -> bool:
        if v not in self.d:
            return False

        e, i = self.a.pop(), self.d.pop(v)
        if i < len(self.a):
            self.a[i], self.d[e] = e, i
        
        return True
        

    def getRandom(self, i=None, duplicates = False) ->  Union[int, List[int]]:
        if not self.a:
            return None

        if i is None:
            return random.choice(self.a)
        else:
            if duplicates:
                return random.choices(self.a, k=i)
            else:
                # Remember, maybe is better to turn it into a list again
                return set(random.choices(self.a, k=i))
        
# def create_graph_from_csv(file_path, add_loops = True):
#     # Create an empty dictionary to store adjacency lists
#     adjacency_lists = {}

#     # Read the CSV file and add edges to the adjacency lists
#     with open(file_path, 'r') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         header = next(csv_reader)  # Skip the header

#         for row in csv_reader:
#             a, b = row
#             a = int(a)
#             b = int(b)

#             # Add edge to the adjacency lists
#             if a not in adjacency_lists:
#                 adjacency_lists[a] = OptList()
#                 if add_loops:
#                     adjacency_lists[a].append(a)
#             adjacency_lists[a].append(b)

#             if b not in adjacency_lists:
#                 adjacency_lists[b] = OptList()
#             if add_loops:
#                     adjacency_lists[b].append(b)
#             adjacency_lists[b].append(a)

#     return adjacency_lists


def create_graph_from_csv(graph, add_loops = True):
    # Create an empty dictionary to store adjacency lists
    adjacency_lists = {}

    # Read the CSV file and add edges to the adjacency lists
        

    for e in graph.edges():
        (a,b) = e
        # Add edge to the adjacency lists
        if a not in adjacency_lists:
            adjacency_lists[a] = OptList()
            if add_loops:
                adjacency_lists[a].append(a)
        adjacency_lists[a].append(b)

        if b not in adjacency_lists:
            adjacency_lists[b] = OptList()
        if add_loops:
                adjacency_lists[b].append(b)
        adjacency_lists[b].append(a)

    return adjacency_lists

# def create_adjlists_from_graph(graph, add_loops = True):
#     # Create an empty dictionary to store adjacency lists
#     adjacency_lists = {}

#     # Read the CSV file and add edges to the adjacency lists
#     for node in graph.nodes():
#         adjacency_lists[node] = [a for a in graph.neighbors(node)]

#     return adjacency_lists

def generate_random_permutation(graph):
    pi_values = {key:random.random() for key in graph.keys()}
    sorted_values = dict(sorted(pi_values.items(), key=lambda item: item[1]))
    # Get a list of nodes from the graph
    #nodes = list(graph.keys())

    # Shuffle the list to create a random node arrival order
    #random.shuffle(nodes)

    return sorted_values


def generate_random_node_order(graph):
    # Get a list of nodes from the graph
    nodes = list(graph.keys())

    # Shuffle the list to create a random node arrival order
    random.shuffle(nodes)

    return nodes


def classical_pivot(graph):
    classical_pivot_clustering = {}
    assigned = set()
    random_node_order = generate_random_node_order(graph)
    random_node_iterator = iter(random_node_order)
    while True:
        try:
            node = next(random_node_iterator)
        except StopIteration:
            break
        if node in assigned:
            continue
        classical_pivot_clustering[node] = node
        assigned.add(node)
        # if not assigned that means it is a pivot
        for neighbor in graph[node]:
            if neighbor in assigned:
                continue
            classical_pivot_clustering[neighbor] = node
            assigned.add(neighbor)
    return classical_pivot_clustering

def connected_components(graph):
    visited = set()
    components = 0
    node_component_dict = {}  # Dictionary to store node -> connected component index

    for node in graph.keys():
        if node not in visited:
            queue = deque([node])
            components+=1
            while queue:
                current_node = queue.popleft()
                if current_node not in visited:
                    visited.add(current_node)
                    node_component_dict[current_node] = components
                    queue.extend(neighbor for neighbor in graph[current_node] if neighbor not in visited)

    return node_component_dict

from collections import defaultdict

def correlation_clustering_value(graph, clustering):
    # Create a dictionary to store the edges between clusters
    num_edges_between_clusters = 0

    # Create a dictionary to store the number of nodes within each cluster
    nodes_in_clusters = defaultdict(int)
    edges_in_clusters = defaultdict(int)

    # Iterate through the edges of the graph
    for node, neighbors in graph.items():
        cluster_id_node = clustering.get(node, None)

        # Increment the number of nodes in the cluster
        nodes_in_clusters[cluster_id_node] += 1
        for neighbor in neighbors:
            cluster_id_neighbor = clustering.get(neighbor, None)
            if cluster_id_node is None or cluster_id_neighbor is None:
                raise ValueError(f"Node ({node}, {neighbor}) has missing cluster ids: ({cluster_id_node},{cluster_id_neighbor})")

            # Check if nodes belong to different clusters
            if cluster_id_node != cluster_id_neighbor:
                # Increment the number of edges between clusters
                num_edges_between_clusters += 1
            else:
                edges_in_clusters[cluster_id_node]+=1
                

    # Calculate the total number of edges between clusters
    num_edges_between_clusters = num_edges_between_clusters/2
    num_non_edges_in_clusters = sum([nodes_in_clusters[cluster_id]*(nodes_in_clusters[cluster_id] -1)/2 - (edges_in_clusters[cluster_id]-nodes_in_clusters[cluster_id])/2 if nodes_in_clusters[cluster_id]> 1 else 0 for cluster_id in edges_in_clusters.keys()]
)        

    return  num_edges_between_clusters + num_non_edges_in_clusters

def ProbAgreement(u, v, g, epsilon, t):
    u_sample = g[u].getRandom(t, True)
    v_sample = g[v].getRandom(t, True)
    Nu_setminus_Nv = sum([w not in g[v] for w in u_sample])
    Nv_setminus_Nu = sum([w not in g[u] for w in v_sample])
    threshold = 0.4*epsilon*t
    return (Nu_setminus_Nv <= threshold) and (Nv_setminus_Nu <= threshold)

def ProbAHeaviness(u, g, epsilon, t):
    # here remember that we may sample ourselves
    u_sample = g[u].getRandom(t, True)
    threshold = 1.2*epsilon*t
    return (sum([not ProbAgreement(u, v, g, epsilon, t) for v in u_sample]) <=threshold)



def one_pivot(current_graph, pi, all_pi):
    #sort adjacency lists
    pivot_list = []
    clusters = {key:[] for key in all_pi.keys()}
    pivot_clustering = {key: key for key in all_pi.keys()}
    eta = {key: key for key in all_pi.keys()}
    for node in current_graph.keys():
        current_graph[node].sort(pi)
        if current_graph[node].a[0] == node:
            pivot_list.append(node)
    for node in current_graph:
        pivot = current_graph[node].a[0]
        eta[node] = pivot
        if pivot in pivot_list:
            clusters[pivot] += [node]
            pivot_clustering[node] = pivot
        else:
            clusters[node] = [node]
            pivot_clustering[node] = node            
    return clusters, pivot_clustering, eta


    
