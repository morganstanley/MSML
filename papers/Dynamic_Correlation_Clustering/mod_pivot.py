import utils
import numpy as np
import random


degree_exploration_threshold = 0.1
epsilon = 0.1
singleton_step = (1+epsilon)

def exploration(node, current_graph, mod_sparse_graph, pi, mod_eta, mod_pivot_clustering, mod_size_clusters, 
clusters_to_check_singleton, singleton_step, mod_clusters, mod_size_last_check):
    for neighbor in current_graph[node]:
        # print(mod_eta[neighbor])
        # if (mod_eta[neighbor] != -1):
        #     print("not one!")
        if (mod_eta[neighbor] != -1):
            if (pi[mod_eta[neighbor]] > pi[node]):
                if mod_eta[neighbor] == neighbor:  ## neighbor is pivot
                    for w in mod_sparse_graph[neighbor]:
                        if w == neighbor:
                            continue
                        if mod_eta[w]==neighbor:
                            if mod_pivot_clustering[w] == neighbor:
                                mod_size_clusters[neighbor] -= 1
                                mod_clusters[neighbor].remove(w)
                            mod_pivot_clustering[w] = w 
                if (mod_pivot_clustering[neighbor] == mod_eta[neighbor]) & (mod_eta[mod_eta[neighbor]] == mod_eta[neighbor]): # for when neighbor is not pivot
                    mod_size_clusters[mod_pivot_clustering[neighbor]] -= 1
                    mod_clusters[mod_pivot_clustering[neighbor]].remove(neighbor)
                mod_size_clusters[neighbor] = 0
                mod_eta[neighbor] = node
                mod_size_clusters[node] += 1
                mod_pivot_clustering[neighbor] = node
                mod_clusters[node] += [neighbor]
        if mod_eta[neighbor] == -1:
            mod_eta[neighbor] = node
            mod_size_clusters[node] += 1
            mod_pivot_clustering[neighbor] = node
            mod_clusters[node] += [neighbor]
    if check_cluster(mod_size_last_check[node], mod_size_clusters[node] , singleton_step):
        clusters_to_check_singleton += [node]
    return mod_eta, mod_size_clusters, mod_pivot_clustering, clusters_to_check_singleton, mod_clusters



def check_cluster(last_check, size, singleton_step):
    if (last_check * (singleton_step) < size) | (last_check/singleton_step > size):
        return True
    return False 

def make_some_nodes_singleton(cluster, mod_current_graph, mod_pivot_clustering, sample_size,  single_threshold=1/2):
    #cluster =[w for w in mod_sparse_graph.keys() if mod_pivot_clustering[w] == pivot]  ## inefficient
    for node in cluster:
        samples = []
        if sample_size > len(cluster):
            samples = cluster
        else:
            samples = random.sample(cluster, sample_size)
        nghbrs = 0
        for sample in samples:
            if sample in mod_current_graph[node]:
                nghbrs +=1
        if nghbrs < len(samples)*single_threshold:
            mod_pivot_clustering[node] = node
    return mod_pivot_clustering

def make_some_nodes_singleton_new(cluster, mod_current_graph, mod_pivot_clustering, sample_size,  single_threshold=1/2):
    #cluster =[w for w in mod_sparse_graph.keys() if mod_pivot_clustering[w] == pivot]  ## inefficient
    for node in cluster:
        if len(mod_current_graph[node])<len(cluster)*single_threshold:
            mod_pivot_clustering[node] = node
        # samples = []
        # if sample_size > len(cluster):
        #     samples = cluster
        # else:
        #     samples = random.sample(cluster, sample_size)
        # nghbrs = 0
        # for sample in samples:
        #     if sample in mod_current_graph[node]:
        #         nghbrs +=1
        # if nghbrs < len(samples)*single_threshold:
        #     mod_pivot_clustering[node] = node
    return mod_pivot_clustering

def modified_pivot_one_step(current_graph, mod_sparse_graph ,mod_eta, mod_pivot_clustering, mod_size_clusters, mod_size_last_check, mod_clusters, pi_values_dict, pi,
 node, threshold1, sample_size, singleton_threshold):
    ## ---------------- our alg ------------------------------------------------------------------------------
    clusters_to_check_singleton = []
    if pi_values_dict[node] <= threshold1/len(current_graph[node]):
        pivot = node
        for neighbor in current_graph[node]:
            mod_sparse_graph[neighbor].append(node)
            mod_sparse_graph[node].append(neighbor)
            if (pi[neighbor]<pi[pivot]):
                pivot = neighbor
        if (pivot == node):
            mod_clusters[node] = [node]
            mod_eta[node] = node
            mod_size_clusters[node] = 1
            mod_size_last_check[node] = 1
            mod_eta, mod_size_clusters, mod_pivot_clustering, clusters_to_check_singleton, mod_clusters = exploration(node, current_graph, mod_sparse_graph, pi, mod_eta, mod_pivot_clustering,
                                                                                                        mod_size_clusters, clusters_to_check_singleton, singleton_step, mod_clusters, mod_size_last_check)
        if (pivot != node):
            mod_eta[node] = pivot
            if (mod_eta[pivot] == pivot): 
                mod_size_clusters[pivot] += 1
                if check_cluster(mod_size_last_check[pivot], mod_size_clusters[pivot], singleton_step):
                    clusters_to_check_singleton += [pivot]
                mod_pivot_clustering[node] = pivot
                mod_clusters[pivot]+= [node]
                if len(current_graph[node]) >= len(current_graph[pivot]) * degree_exploration_threshold:
                    mod_eta, mod_size_clusters, mod_pivot_clustering, clusters_to_check_singleton, mod_clusters = exploration(pivot, current_graph, mod_sparse_graph, pi, mod_eta, mod_pivot_clustering, 
                                                                                                                mod_size_clusters, clusters_to_check_singleton, singleton_step, mod_clusters, mod_size_last_check)
                else:
                    mod_pivot_clustering[node] = node
    else:
        #print('samples', sample_size, len(current_graph[node]))
        samples = []
        if len(current_graph[node]) < sample_size:
            samples = current_graph[node]
        else:
            samples = current_graph[node].getRandom(sample_size)
        pivot = list(samples)[0]
        for sample in samples:
            if sample not in mod_sparse_graph[node]:
                mod_sparse_graph[sample].append(node)
                mod_sparse_graph[node].append(sample)
            if pi[pivot] > pi[sample]:
                pivot = sample 
            if pivot in current_graph[node]: ## is this query in linear time?
                if pivot not in mod_sparse_graph[node]:
                    mod_sparse_graph[pivot].append(node)
                    mod_sparse_graph[node].append(pivot)                       
                if mod_eta[sample] != -1:
                    if pi[mod_eta[sample]] < pi[pivot]:
                        pivot = mod_eta[sample] 
        if (mod_eta[pivot] == pivot):
            mod_eta[node] = pivot
            if mod_pivot_clustering[node] in mod_size_clusters:
                mod_size_clusters[mod_pivot_clustering[node]] -= 1
            mod_pivot_clustering[node] = mod_eta[node]
            mod_clusters[mod_eta[node]] += [node]
            if mod_eta[node] in mod_size_clusters:
                mod_size_clusters[mod_eta[node]] += 1
                if check_cluster(mod_size_last_check[mod_eta[node]], mod_size_clusters[mod_eta[node]], singleton_step):
                    clusters_to_check_singleton += [mod_eta[node]]
            else:
                mod_size_clusters[mod_eta[node]] = 1 
        else:
            mod_pivot_clustering[node] = node
                
    ##### --------------- singleton 
    for pivot in clusters_to_check_singleton:
        mod_pivot_clustering = make_some_nodes_singleton(mod_clusters[pivot], current_graph, mod_pivot_clustering, sample_size, singleton_threshold)
        mod_size_last_check[pivot] = mod_size_clusters[pivot]
    return mod_eta, mod_pivot_clustering, mod_size_clusters, mod_size_last_check, mod_clusters, mod_sparse_graph



def pivot_recompute(current_graph, pi, pi_values_dict, threshold1, mod_sparse_graph, mod_eta, mod_pivot_clustering):
    clusters = {}
    for node in current_graph.keys():
        if pi_values_dict[node] <= threshold1/len(current_graph[node]):
            pivot = node
            for neighbor in current_graph[node]:
                mod_sparse_graph[neighbor].append(node)
                mod_sparse_graph[node].append(neighbor)
                if pi[neighbor]<pi[pivot]:
                    pivot = neighbor
            if pivot == node:
                clusters[node] = [node]
                mod_pivot_clustering[node] = node
                mod_eta[node] = node
                for neighbor in current_graph[node]:
                    if (mod_eta[neighbor] == -1):
                        mod_eta[neighbor] = node
                        mod_pivot_clustering[neighbor] = node
                    elif (pi[mod_eta[neighbor]] > pi[node]):
                        mod_eta[neighbor] = node
                        mod_pivot_clustering[neighbor] = node
    for node in current_graph:
        if (mod_eta[node] != -1) & (mod_eta[node] != node):
            if mod_eta[node] != mod_eta[mod_eta[node]]:
                print('scream')
            clusters[mod_eta[node]] += [node]
    return clusters, mod_pivot_clustering, mod_eta, mod_sparse_graph
