import math
import numpy as np
import importlib

import spectral_clusters_optimizer as sco
importlib.reload(sco)

import kmeans_clusters_optimizer as kco
importlib.reload(kco)

import itc_mst_clusters_optimizer as itc_mst
importlib.reload(itc_mst)

import itc_knn_clusters_optimizer as itc_knn
importlib.reload(itc_knn)


# wrapper class to run the most essential clustering baselines
class ClusteringEvaluator:

    def __init__(self, num_clusters=2, seed=0):

        self.num_clusters = num_clusters
        self.seed = seed

    def cluster_labels_optimize(self, data, cluster_algo):

        if cluster_algo == 'spectral':
            clusters = sco.SpectralClusterOptimizer(num_clusters=self.num_clusters, seed=self.seed).optimize(data=data)
        elif cluster_algo == 'kmeans':
            clusters = kco.KMeansClusterOptimizer(num_clusters=self.num_clusters, seed=self.seed).optimize(data=data)
        elif cluster_algo == 'itc_mst':
            clusters = itc_mst.ITCOptimizer(num_clusters=self.num_clusters, seed=self.seed).optimize(data=data)
        elif cluster_algo == 'itc_knn':
            clusters = itc_knn.ITCOptimizer(num_clusters=self.num_clusters, seed=self.seed).optimize(data=data)
        else:
            raise NotImplemented

        return clusters
