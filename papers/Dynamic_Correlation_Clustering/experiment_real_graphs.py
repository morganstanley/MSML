"""
Dynamic correlation clustering experiments on real-world graph datasets.

Compares three algorithms:
  - DYNAMIC-AGREEMENT (Cohen-Addad et al.)
  - PIVOT-DYNAMIC (reference pivot-based clustering)
  - SPARSE-PIVOT (our algorithm, from mod_pivot.py)

Datasets: Facebook, Email-Enron, Cit-HepTh, CA-AstroPh
Outputs: one PDF plot per dataset showing normalized clustering objective over time.
"""
import random
import time
import warnings
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import mod_pivot as mpiv
from utils import (OptList, create_graph_from_csv, generate_random_node_order,
                   generate_random_permutation, classical_pivot,
                   connected_components, correlation_clustering_value,
                   ProbAgreement, ProbAHeaviness)

warnings.filterwarnings('ignore', 'divide by zero')

# ── Parameters ────────────────────────────────────────────────────────────────
ANCHOR_PROBABILITY_NUMERATOR = 20
NOTIFICATIONS = 2
EPSILON = 0.2
DELETION_PROB = 0.1
NUM_AGREEMENT_SAMPLES = 2
NUM_CONNECT_SAMPLES = 2
MOD_EPSILON = 0.1
SINGLETON_THRESHOLD = 0.5
NUM_EXPERIMENTS = 1
SNAPSHOT_INTERVAL = 50


def load_datasets():
    """Load the four real-world graph datasets."""
    paths = [
        "datasets/musae_facebook.csv",
        "datasets/email-Enron.txt.gz",
        "datasets/cit-HepTh.txt.gz",
        "datasets/ca-AstroPh.txt.gz",
    ]
    names = ["facebook", "email-Enron", "cit-HepTh", "ca-AstroPh"]
    graphs = [nx.read_edgelist(paths[0], delimiter=',', nodetype=int)]
    for path in paths[1:]:
        graphs.append(nx.read_edgelist(path, nodetype=int))
    return list(zip(names, graphs))


def take_snapshot(current_graph, sparse_graph, pivot_clustering,
                  mod_pivot_clustering, mod_nodes_present):
    """Compute normalized clustering objectives for all algorithms."""
    clustering = connected_components(sparse_graph)
    all_singletons = {node: idx for idx, node in enumerate(sparse_graph.keys())}
    pivot_step = {u: pivot_clustering[u] for u in sparse_graph.keys()}
    mod_step = {u: mod_pivot_clustering[u]
                for u in mod_nodes_present if u in current_graph}

    corr_clustering = correlation_clustering_value(current_graph, clustering)
    corr_singletons = correlation_clustering_value(current_graph, all_singletons)
    corr_pivot = correlation_clustering_value(current_graph, pivot_step)
    corr_mod = correlation_clustering_value(current_graph, mod_step)

    if corr_singletons == 0:
        return None
    return (corr_clustering / corr_singletons,
            1.0,
            corr_pivot / corr_singletons,
            corr_mod / corr_singletons)


def run_experiment(name, graph):
    """Run the dynamic clustering experiment on a single graph."""
    print(f"\n{'='*70}", flush=True)
    print(f"Dataset: {name} ({len(graph.nodes())} nodes, {len(graph.edges())} edges)",
          flush=True)

    adjacency_lists = create_graph_from_csv(graph)
    num_nodes = len(graph.nodes())
    threshold1 = np.log(num_nodes)
    sample_size = int(np.floor(np.log(num_nodes)))

    # Result lists
    vals_agreement = []
    vals_singletons = []
    vals_pivot = []
    vals_mod = []

    for t in range(NUM_EXPERIMENTS):
        random.seed(t)
        random_node_order = generate_random_node_order(adjacency_lists)

        # ── Shared random permutation ──
        pi_values_dict = generate_random_permutation(adjacency_lists)
        pi = {node: idx for idx, node in enumerate(pi_values_dict.keys())}

        # ── PIVOT-DYNAMIC state ──
        eta = {k: k for k in pi}
        pivot_clustering = {k: k for k in pi}

        # ── DYNAMIC-AGREEMENT state ──
        sparse_graph = {}
        nodes_present = OptList()
        Phi = set()
        Phi_nodes = {}
        I_nodes = defaultdict(set)
        B_nodes = {node: OptList() for node in adjacency_lists}
        current_graph = {}

        # ── SPARSE-PIVOT state ──
        mod_pi_values_dict = dict(pi_values_dict)
        mod_pi = {node: idx for idx, node in enumerate(mod_pi_values_dict.keys())}
        mod_eta = {k: -1 for k in pi}
        mod_pivot_clustering = {k: k for k in pi}
        mod_size_clusters = {}
        mod_size_last_check = {k: 1 for k in pi}
        mod_sparse_graph = {}
        mod_clusters = {k: [] for k in pi}
        mod_current_graph = {}
        mod_nodes_present = OptList()
        num_deletions = 0

        # ── Timing ──
        time_pivot = 0.0
        time_agreement = 0.0
        time_mod = 0.0

        worst_pivot = 0.0
        worst_agreement = 0.0
        worst_mod = 0.0

        time_agreement_add = 0.0
        time_agreement_del = 0.0
        time_mod_add = 0.0
        time_mod_del = 0.0

        worst_agreement_add = 0.0
        worst_agreement_del = 0.0
        worst_mod_add = 0.0
        worst_mod_del = 0.0

        random_node_iterator = iter(random_node_order)
        step = -1
        only_deletion = False

        while True:
            step += 1
            if (step > 100) and (not current_graph):
                break

            is_deletion = bool(((random.random() < DELETION_PROB) and (step > 0)
                                and current_graph)
                               or (only_deletion and current_graph))

            if is_deletion:
                node = nodes_present.getRandom()
                neighbors = list(current_graph[node])

                # ── PIVOT-DYNAMIC deletion ──
                t0 = time.time()
                for nb in neighbors:
                    current_graph[nb].remove(node)
                    if node == nb:
                        continue
                    if (eta[nb] != node) or (eta[node] != nb):
                        continue
                    if eta[nb] == node:
                        eta[nb] = nb
                        for w in current_graph[nb]:
                            if pi[w] < pi[eta[nb]]:
                                eta[nb] = w
                        if eta[nb] == nb:
                            pivot_clustering[nb] = nb
                            for w in current_graph[nb]:
                                if w != nb:
                                    pivot_clustering[w] = (nb if eta[w] == nb
                                                           else pivot_clustering[w])
                        else:
                            pivot_clustering[nb] = (eta[nb] if eta[nb] == eta[eta[nb]]
                                                    else nb)
                    else:
                        eta[node] = node
                        for w in current_graph[node]:
                            if pi[w] < pi[eta[node]]:
                                eta[node] = w
                        if eta[node] == node:
                            pivot_clustering[node] = node
                            for w in current_graph[node]:
                                if w != node:
                                    pivot_clustering[w] = (node if eta[w] == node
                                                           else pivot_clustering[w])
                        else:
                            pivot_clustering[node] = (eta[node]
                                                      if eta[node] == eta[eta[node]]
                                                      else node)
                step_time = time.time() - t0
                time_pivot += step_time
                worst_pivot = max(worst_pivot, step_time)
                nodes_present.remove(node)
                del current_graph[node]

                # ── SPARSE-PIVOT deletion ──
                t0 = time.time()
                num_deletions += 1
                if num_deletions > MOD_EPSILON * num_nodes:
                    num_deletions = 0
                    mod_pi_values_dict = generate_random_permutation(adjacency_lists)
                    mod_pi = {node: idx for idx, node in enumerate(mod_pi_values_dict.keys())}
                    mod_nodes_present = nodes_present
                    mod_eta = {k: -1 for k in mod_pi}
                    mod_pivot_clustering = {k: k for k in mod_pi}
                    mod_sparse_graph = {k: OptList() for k in current_graph}
                    (mod_clusters, mod_pivot_clustering, mod_eta,
                     mod_sparse_graph) = mpiv.pivot_recompute(
                        current_graph, mod_pi, mod_pi_values_dict,
                        threshold1, mod_sparse_graph, mod_eta, mod_pivot_clustering)
                    mod_current_graph = {k: current_graph[k] for k in current_graph}
                    mod_size_clusters = {k: len(mod_clusters[k]) for k in mod_clusters}
                    mod_size_last_check = dict(mod_size_clusters)
                    for piv in mod_clusters:
                        if len(mod_clusters[piv]) > 2:
                            mod_pivot_clustering = mpiv.make_some_nodes_singleton(
                                mod_clusters[piv], mod_sparse_graph,
                                mod_pivot_clustering, sample_size, SINGLETON_THRESHOLD)
                step_time = time.time() - t0
                time_mod += step_time
                worst_mod = max(worst_mod, step_time)
                time_mod_del += step_time
                worst_mod_del = max(worst_mod_del, step_time)

            else:
                try:
                    node = next(random_node_iterator)
                except StopIteration:
                    only_deletion = True
                    continue

                # ── PIVOT-DYNAMIC addition ──
                current_graph[node] = OptList()
                nodes_present.append(node)
                for nb in adjacency_lists[node]:
                    if nb in nodes_present:
                        t0 = time.time()
                        if pi[eta[nb]] > pi[node]:
                            if eta[nb] == nb:
                                for w in current_graph[nb]:
                                    if w != nb:
                                        pivot_clustering[w] = (w if eta[w] == nb
                                                               else pivot_clustering[w])
                            eta[nb] = node
                            pivot_clustering[nb] = (node if eta[node] == node else nb)
                        if pi[eta[node]] > pi[nb]:
                            if eta[node] == node:
                                for w in current_graph[node]:
                                    if w != node:
                                        pivot_clustering[w] = (w if eta[w] == node
                                                               else pivot_clustering[w])
                            eta[node] = nb
                            pivot_clustering[node] = (nb if eta[nb] == nb else node)
                        step_time = time.time() - t0
                        time_pivot += step_time
                        worst_pivot = max(worst_pivot, step_time)
                        current_graph[nb].append(node)
                        current_graph[node].append(nb)

                # ── SPARSE-PIVOT addition ──
                mod_current_graph[node] = OptList()
                for nb in set(adjacency_lists[node]).intersection(set(mod_nodes_present)):
                    mod_current_graph[node].append(nb)
                    mod_current_graph[nb].append(node)
                mod_nodes_present.append(node)
                mod_sparse_graph[node] = OptList()
                t0 = time.time()
                (mod_eta, mod_pivot_clustering, mod_size_clusters,
                 mod_size_last_check, mod_clusters,
                 mod_sparse_graph) = mpiv.modified_pivot_one_step(
                    mod_current_graph, mod_sparse_graph, mod_eta,
                    mod_pivot_clustering, mod_size_clusters, mod_size_last_check,
                    mod_clusters, mod_pi_values_dict, mod_pi, node,
                    threshold1, sample_size, SINGLETON_THRESHOLD)
                step_time = time.time() - t0
                time_mod += step_time
                worst_mod = max(worst_mod, step_time)
                time_mod_add += step_time
                worst_mod_add = max(worst_mod_add, step_time)

            # ── DYNAMIC-AGREEMENT notifications ──
            t0 = time.time()
            notified = set()
            skip_agreement_body = False

            if is_deletion:
                for v in I_nodes[node]:
                    B_nodes[v].remove(node)
                got_type_0 = B_nodes[node].getRandom(NOTIFICATIONS)
                if not got_type_0:
                    skip_agreement_body = True
                else:
                    got_type_0.discard(node)
                    if not got_type_0:
                        skip_agreement_body = True
            else:
                notified.add(node)
                got_type_0 = current_graph[node].getRandom(NOTIFICATIONS)
                if got_type_0:
                    got_type_0.discard(node)
                    I_nodes[node].update(got_type_0)
                    for v in got_type_0:
                        B_nodes[v].append(node)

            if not skip_agreement_body:
                # Type 1 notifications
                got_type_1 = set()
                for v in got_type_0:
                    if v not in current_graph:
                        continue
                    for u in I_nodes[v]:
                        B_nodes[u].remove(v)
                    I_nodes[v].update(current_graph[v].getRandom(NOTIFICATIONS))
                    I_nodes[v].discard(v)
                    I_nodes[v].discard(node)
                    for u in I_nodes[v]:
                        B_nodes[u].append(v)
                        if u not in notified:
                            got_type_1.add(u)
                            notified.add(u)

                # Type 2 notifications
                got_type_2 = set()
                for v in got_type_1:
                    if v not in current_graph:
                        continue
                    for u in I_nodes[v]:
                        B_nodes[u].remove(v)
                    I_nodes[v].update(current_graph[v].getRandom(NOTIFICATIONS))
                    I_nodes[v].discard(v)
                    I_nodes[v].discard(node)
                    for u in I_nodes[v]:
                        B_nodes[u].append(v)
                        if u not in notified or u not in got_type_2:
                            got_type_2.add(u)

                # Receive Type 2
                for v in got_type_2:
                    if v not in current_graph:
                        continue
                    for u in I_nodes[v]:
                        B_nodes[u].remove(v)
                    I_nodes[v] = current_graph[v].getRandom(NOTIFICATIONS)
                    I_nodes[v].discard(v)
                    I_nodes[v].discard(node)
                    for u in I_nodes[v]:
                        B_nodes[u].append(v)

                # Sparse graph bookkeeping
                if not is_deletion:
                    sparse_graph[node] = OptList()
                    Phi_nodes[node] = set()
                else:
                    Phi.discard(node)
                    for nb in list(sparse_graph[node]):
                        sparse_graph[nb].remove(node)
                        sparse_graph[node].remove(nb)
                        Phi_nodes[nb].discard(node)
                    del sparse_graph[node]
                    del Phi_nodes[node]

                # Anchor / Clean / Connect for each notified node
                for u in notified:
                    if is_deletion and node == u:
                        continue
                    if u not in nodes_present:
                        continue
                    if u in Phi:
                        for v in list(sparse_graph[u]):
                            if (u == v) or (v in Phi):
                                continue
                            sparse_graph[u].remove(v)
                            sparse_graph[v].remove(u)
                            Phi_nodes[v].discard(u)
                    Phi.discard(u)

                    anchor_prob = ANCHOR_PROBABILITY_NUMERATOR / len(current_graph[u])
                    if ((random.random() < anchor_prob) and
                            ProbAHeaviness(u, current_graph, EPSILON, NUM_AGREEMENT_SAMPLES)):
                        Phi.add(u)
                        for v in current_graph[u]:
                            if ProbAgreement(u, v, current_graph, EPSILON, NUM_AGREEMENT_SAMPLES):
                                sparse_graph[u].append(v)
                                sparse_graph[v].append(u)
                                if u != v:
                                    Phi_nodes[v].add(u)

                    for w in list(Phi_nodes[u]):
                        if w not in nodes_present:
                            Phi_nodes[u].discard(w)
                            continue
                        if w == u:
                            continue
                        if (ProbAgreement(u, w, current_graph, EPSILON, NUM_AGREEMENT_SAMPLES) and
                                ProbAHeaviness(w, current_graph, EPSILON, NUM_AGREEMENT_SAMPLES)):
                            continue
                        sparse_graph[u].remove(w)
                        sparse_graph[w].remove(u)
                        Phi_nodes[u].discard(w)

                    for w in current_graph[u].getRandom(NUM_CONNECT_SAMPLES):
                        for r in list(Phi_nodes[w]):
                            if r not in nodes_present:
                                Phi_nodes[w].discard(r)
                                continue
                            if r not in current_graph[u]:
                                continue
                            if (ProbAgreement(u, r, current_graph, EPSILON, NUM_AGREEMENT_SAMPLES) and
                                    ProbAHeaviness(r, current_graph, EPSILON, NUM_AGREEMENT_SAMPLES)):
                                sparse_graph[u].append(r)
                                sparse_graph[r].append(u)
                                Phi_nodes[u].add(r)
            else:
                # Deletion with empty notification: still maintain sparse graph
                if is_deletion:
                    Phi.discard(node)
                    for nb in list(sparse_graph[node]):
                        sparse_graph[nb].remove(node)
                        sparse_graph[node].remove(nb)
                        Phi_nodes[nb].discard(node)
                    del sparse_graph[node]
                    del Phi_nodes[node]

            step_time = time.time() - t0
            time_agreement += step_time
            worst_agreement = max(worst_agreement, step_time)
            if is_deletion:
                time_agreement_del += step_time
                worst_agreement_del = max(worst_agreement_del, step_time)
            else:
                time_agreement_add += step_time
                worst_agreement_add = max(worst_agreement_add, step_time)

            # ── Snapshot ──
            if (step + 1) % SNAPSHOT_INTERVAL == 0:
                result = take_snapshot(current_graph, sparse_graph,
                                       pivot_clustering, mod_pivot_clustering,
                                       mod_nodes_present)
                if result is not None:
                    vals_agreement.append(result[0])
                    vals_singletons.append(result[1])
                    vals_pivot.append(result[2])
                    vals_mod.append(result[3])
                if (step + 1) % (SNAPSHOT_INTERVAL * 20) == 0:
                    print(f"  step {step+1}", flush=True)

    ne = float(NUM_EXPERIMENTS)
    print(f"  DYNAMIC-AGREEMENT: {np.mean(vals_agreement):.4f}  "
          f"PIVOT-DYNAMIC: {np.mean(vals_pivot):.4f}  "
          f"SPARSE-PIVOT: {np.mean(vals_mod):.4f}", flush=True)
    print(f"  Total time — agreement: {time_agreement/ne:.2f}s  "
          f"pivot: {time_pivot/ne:.2f}s  mod: {time_mod/ne:.2f}s", flush=True)
    print(f"  Worst update — agreement: {worst_agreement:.4f}s  "
          f"pivot: {worst_pivot:.4f}s  mod: {worst_mod:.4f}s", flush=True)
    print(f"  Agreement additions — total: {time_agreement_add/ne:.2f}s  "
          f"worst: {worst_agreement_add:.4f}s", flush=True)
    print(f"  Agreement deletions — total: {time_agreement_del/ne:.2f}s  "
          f"worst: {worst_agreement_del:.4f}s", flush=True)
    print(f"  Mod additions — total: {time_mod_add/ne:.2f}s  "
          f"worst: {worst_mod_add:.4f}s", flush=True)
    print(f"  Mod deletions — total: {time_mod_del/ne:.2f}s  "
          f"worst: {worst_mod_del:.4f}s", flush=True)

    # ── Plot ──
    if vals_agreement:
        sns.set_palette("colorblind")
        fig, ax = plt.subplots()
        ax.plot(vals_agreement, label='DYNAMIC-AGREEMENT')
        ax.plot(vals_singletons, label='SINGLETONS')
        ax.plot(vals_pivot, label='PIVOT-DYNAMIC')
        ax.plot(vals_mod, label='SPARSE-PIVOT')
        ax.set_title(name)
        ax.set_xlabel('Node Arrivals/Deletions')
        ax.set_ylabel('Relative Clustering Objective')
        ax.legend()
        plt.savefig(f"{name}.pdf", format="pdf")
        plt.close()
        print(f"  Plot saved: {name}.pdf", flush=True)


def main():
    datasets = load_datasets()
    for name, graph in datasets:
        run_experiment(name, graph)
    print("\nAll experiments complete.", flush=True)


if __name__ == '__main__':
    main()
