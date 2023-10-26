import clusters_optimizer_base as co
from sklearn.cluster import KMeans


class KMeansClusterOptimizer(co.ClustersOptimizerBase):

    # 100 initializations

    def optimize(self, data):
        obj = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.seed,
            n_init=100,
        )
        obj.fit(data)
        return obj.labels_
