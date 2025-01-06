# Reweighting Improves Conditional Risk Bounds
This is the official code repository for the paper titled **"Reweighting Improves Conditional Risk Bounds"** [(link to paper)](https://openreview.net/pdf?id=MvYddudHuE), accepted in _Transaction on Machine Learning Research_ (TMLR), 2024. 

Authors: Yikai Zhang, Jiahe Lin, Fengpei Li, Songzhu Zheng, Anant Raj, Anderson Schneider, Yuriy Nevmyvaka. 

## Abstract
In this work, we study the weighted empirical risk minimization (weighted ERM) schema, in which an additional data-dependent weight function is incorporated when the empirical risk function is being minimized. 
We show that under a general "balanceable" Bernstein condition, one can design a weighted ERM estimator to achieve superior performance in certain sub-regions over the one obtained from standard ERM, 
and the superiority manifests itself through a data-dependent constant term in the error bound. 
These sub-regions correspond to large-margin ones in classification settings and low-variance ones in heteroscedastic regression settings, respectively. 
Our findings are supported by evidence from synthetic data experiments.


## Environment Setup

The following installs all the dependencies
```console
pip install -e .
```

## Synthetic Data Experiments

This section provides instructions on how to setup/run the experiments reported in Section 5 of the manuscript. 

- Data generation:
    - classification setting:
        ```console
        ./bin/prep-clsf-data --ds-str=ds_clsf --view-dataset
        ```
    - regression setting:
        ```console
        ./bin/prep-regr-data --ds-str=ds_regr --view-dataset
        ```
- Run experiments on a specific synthetic dataset using neural network:
    ```console
    ./bin/train-sim --ds-str=ds_regr --cuda=0 --n-replica=1 --train-size=20000
    ```

## Citation
```
@article{zhang2024reweighting,
    title={Reweighting Improves Conditional Risk Bounds},
    author={Zhang, Yikai and Lin, Jiahe and Li, Fengpei and Zheng, Songzhu and Schneider, Anderson and Nevmyvaka, Yuriy and Raj, Anat},
    journal={Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=MvYddudHuE},
}
```

## License
All source files in this repository, unless explicitly mentioned otherwise, are released under the Apache 2.0 license, the text of which can be found in the LICENSE file.

## Contact
Authors: yikai.zhang@morganstanley.com; jiahe.lin@morganstanley.com

Morgan Stanley Machine Learning Research: msml-qa@morganstanley.com
