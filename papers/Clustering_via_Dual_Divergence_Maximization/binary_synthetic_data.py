import numpy as np
import numpy.random as npr


class BinarySyntheticData:
    # binary data allows modeling complex dependencies between input dimensions     

    def sample_timeseries_with_dependencies(self, n=10000, d=10, p_seed=0, seed=0):
        
        # generate d number of timeseries which are dependent w.r.t. each other         
        # n is number of samples/observations in a timeseries         
        
        # data sampler         
        data_sampler = npr.RandomState(seed=seed)
        # probability sampler
        p_sampler = npr.RandomState(seed=p_seed)
        
        # d timeseries with n observations in each
        # observations within timeseries are treated i.i.d., we model dependence between timeseries only         
        X = np.zeros((n, d), dtype=bool)
        
        for i in range(d):

            if i == 0:
                # sample from marginal for the first dimension                 
                X[data_sampler.choice(n, int(0.5*n), replace=False), i] = True
                continue
            
            # binary codes to condition upon for generating samples of next dimension             
            _, sample_ids = np.unique(X[:, :i], axis=0, return_inverse=True)
            unique_ids = np.unique(sample_ids)
            for id in unique_ids:
                
                # idx of samples with unique binary code                 
                data_idx = np.where(sample_ids == id)[0]
                # sample prob number (0, 1) for the number of ones                 
                q = p_sampler.uniform(low=0.0, high=1.0)
                
                # sample ones for current dimension                 
                ones_idx = data_sampler.choice(
                    data_idx.size,
                    int(q*data_idx.size),
                    replace=False,
                )
                X[data_idx[ones_idx], i] = True
                data_idx, ones_idx = (None,)*2
        
        return X

    def sample_from_dependencies(self, n=3000, num_clusters=2, num_timeseries_per_cluster=10):
        # n is number of samples
        
        assert num_clusters > 1
        
        # total dimensions d correspond to input size for clustering
        # think of each dimenion as a univariate timeseries         
        # number of samples n corresponds to dimension of the input space in which clustering is performed
        # same as in a timeseries, number of observations are treated as dimension of input space when clustering timeseries          
        # we have typically a scenario of high dimensional input space for clustering with very few samples as input for clustering         
        d = num_timeseries_per_cluster*num_clusters
        
        # initialization of data array         
        X = np.zeros((n, d), dtype=bool)
        # cluster labels         
        labels = np.ones(d)
        for cluster in range(num_clusters):          
            # all the dimensions (timeseries) within a cluster are dependent upon each other          
            X[:, cluster*num_timeseries_per_cluster:(cluster+1)*num_timeseries_per_cluster] = self.sample_timeseries_with_dependencies(n=n, d=num_timeseries_per_cluster, p_seed=cluster, seed=cluster)
            labels[cluster*num_timeseries_per_cluster:(cluster+1)*num_timeseries_per_cluster] = cluster + 1

        # shuffle the order of dimensions (timeseries)
        shuffled_idx = npr.RandomState(seed=0).choice(X.shape[1], size=X.shape[1], replace=False)
        X = X[:, shuffled_idx]
        labels = labels[shuffled_idx]

        return X, labels
