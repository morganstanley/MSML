import numpy as np


def uniform_distribute_tasks_across_cores(num_tasks, num_cores):
    range_parallel = [np.arange(curr_core, num_tasks, num_cores, dtype=np.int) for curr_core in range(num_cores)]
    return range_parallel
