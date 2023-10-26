import importlib
from sklearn.cluster import SpectralClustering

import clusters_optimizer_base as co
importlib.reload(co)


class SpectralClusterOptimizer(co.ClustersOptimizerBase):

    # 100 initializations
    def optimize(self, data):
        obj = SpectralClustering(
            n_clusters=self.num_clusters,
            assign_labels='discretize',
            random_state=self.seed,
            n_init=100,
        )
        obj.fit(data)
        return obj.labels_
