Code repository: https://github.com/AlbertYuChen/point_process_coupling_public

A simple demo is in `Code/Simple_Example.ipynb` which contains
synthetic data generation and model fitting.

`Code/Data_Generator_Model_Fitting.ipynb` presents more examples.

We provide some tools for neuroscience data, including data download
script, data exploration, data preprocessing and model fitting. The
scripts are under `Code/neuroscience`.


Reference:
Short-term Temporal Dependency Detection under Heterogeneous Event
Dynamic with Hawkes Processes Yu Chen, Fengpei Li, Anderson Schneider,
Yuriy Nevmyvaka, Asohan Amarasingham, Henry Lam, Conference on
Uncertainty in Artificial Intelligence (UAI), 2023 **Oral
presentation**

Abstract: Many *event sequence* data exhibit mutually exciting or
inhibiting patterns. Reliable detection of such temporal dependency is
crucial for scientific investigation. The *de facto* model is the
Multivariate Hawkes Process (MHP), whose impact function naturally
encodes a causal structure in Granger causality. However, the vast
majority of existing methods use direct or nonlinear transform of
*standard* MHP intensity with constant baseline, inconsistent with
real-world data. Under irregular and unknown heterogeneous intensity,
capturing temporal dependency is hard as one struggles to distinguish
the effect of mutual interaction from that of intensity
fluctuation. In this paper, we address the short-term temporal
dependency detection issue. We show the maximum likelihood estimation
(MLE) for cross-impact from MHP has an error that can not be
eliminated but may be reduced by order of magnitude, using
heterogeneous intensity not of the target HP but of the interacting
HP. Then we proposed a robust and computationally-efficient method
modified from MLE that does not rely on the prior estimation of the
heterogeneous intensity and is thus applicable in a data-limited
regime (e.g., few-shot, no repeated observations). Extensive
experiments on various datasets show that our method outperforms
existing ones by notable margins, with highlighted novel applications
in neuroscience.
