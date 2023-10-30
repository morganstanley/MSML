

class ClustersOptimizerBase:

    def __init__(self, num_clusters=2, seed=0, is_debug=False):
        self.num_clusters = num_clusters
        self.seed = seed
        self.is_debug = is_debug
