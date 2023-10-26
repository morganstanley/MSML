import os
# parallel computing is done using multiprocessing in python code layer instead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import importlib
import multiprocessing
import numpy as np

import npeets
importlib.reload(npeets)
# todo: implement a logger


class TimeseriesDependencyScores:

    def __init__(self, num_cores=1, metric='mi', normalize=False):

        assert metric in ['mi', 'mid']
        # metric for measuring dependency between a pair of timeseries
        # samples within a timeseries are treated as i.i.d.
        # 'mid' refers to estimating mutual information between two timeseries with discrete observations.
        # whereas 'mi' refers to estimating mutual information between two timeseries with continuous valued observations.
        self.metric = metric
        # number of cores to use for parallelizing computation of the metric.
        # As an example, for 1000 timeseries, if there are 1 million times that dependency score is computed for a given pair of timeseries.
        # If timeseries has large number of observations, it can be expensive to compute.
        # So, parallel computing is useful and straightforward.
        self.num_cores = num_cores
        self.normalize = normalize
        if normalize:
            assert metric == 'mid'

    def compute_dependency_score_wrapper(self, i):

        num_time_series = self.time_series_data.shape[0]
        scores = np.zeros(num_time_series)

        if self.normalize:
            assert self.metric == 'mid'
            # for normalizing mutual information metric
            # mutual information between discrete random variables can be particularly low in magnitude, especially in the case of long tailed distribution. So, it is useful to normalize for a better interpretability of the metric.
            e_i = npeets.entropyd(sx=self.time_series_data[i], base=2)

        # somewhat crude yet efficient way of monitoring progress
        print('.', end='')

        for j in range(num_time_series):

            if self.metric == 'mi':
                assert not self.normalize
                # k=3 is supposed to be optimal in most cases when estimating mutual information between two 1-D random variables corresponding from two univariate timeseries with observations treated as i.i.d.
                # k = 5 should also be fine
                mi_ij = npeets.mi(x=self.time_series_data[i], y=self.time_series_data[j], k=3, base=2, alpha=0)
                scores[j] = mi_ij
            elif self.metric == 'mid':
                # discrete mutual information
                mi_ij = npeets.midd(x=self.time_series_data[i], y=self.time_series_data[j], base=2)

                if self.normalize:
                    e_j = npeets.entropyd(sx=self.time_series_data[j], base=2)
                    if (e_i + e_j) == 0:
                        assert mi_ij == 0
                    else:
                        mi_ij = 2*mi_ij/(e_i+e_j)

                scores[j] = mi_ij
            else:
                raise NotImplemented

        return scores

    def compute_dependency_scores(self, time_series_data):

        num_time_series, num_timesteps  = time_series_data.shape
        print(num_time_series, num_timesteps)

        self.time_series_data = time_series_data
        dependency_scores = np.zeros((num_time_series, num_time_series), dtype=np.float)

        if self.num_cores == 1:
            for i in range(num_time_series):
                dependency_scores[i] = self.compute_dependency_score_wrapper(i)
        else:
            assert self.num_cores > 1
            with multiprocessing.Pool(processes=min(self.num_cores, num_time_series)) as pool:
                results = pool.map(self.compute_dependency_score_wrapper, np.arange(num_time_series, dtype=np.int))

            for i in range(num_time_series):
                dependency_scores[i] = results[i]
                print('*', end='')
            print('')

        self.time_series_data = None

        return dependency_scores
