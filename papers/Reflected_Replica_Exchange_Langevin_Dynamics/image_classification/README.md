# Reflected Replica Exchange Stochastic Gradient Langevin Dynamics

Please cite our paper if you find it useful in uncertainty estimations

```
@article{zheng2024constrained,
  title={Constrained Exploration via Reflected Replica Exchange Stochastic Gradient Langevin Dynamics},
  author={Zheng, Haoyang and Du, Hengrong and Feng, Qi and Deng, Wei and Lin, Guang},
  journal={arXiv preprint arXiv:2405.07839},
  year={2024}
}
```

## Requirement
Please refer to "env_image_classification.yml"

## ResNet on CIFAR100
For SGDM, please run:
```python
$ python bayes_cnn.py -T 0 -if_domain False
```

For R-SGDM, please run:
```python
$ python bayes_cnn.py -T 0 -if_domain True
```

For SGHMC, please run:
```python
$ python bayes_cnn.py -if_domain False
```

For R-SGHMC, please run:
```python
$ python bayes_cnn.py -if_domain True
```

For cycSGHMC, please run:
```python
$ python bayes_cnn.py -optimizer csgld -lr 4e-5 -if_domain False
```

For R-cycSGHMC, please run:
```python
$ python bayes_cnn.py -optimizer csgld -lr 4e-5 -if_domain True
```

For reSGHMC, please run:
```python
$ python bayes_cnn.py -chains 4 -if_domain False
```

For r2SGHMC, please run:
```python
$ python bayes_cnn.py -chains 4 -if_domain True
```

## Results
![image](https://github.com/haoyangzheng1996/r2SGLD/assets/38525155/94c0f090-f80d-47f8-986a-ecdb0c5fa9aa)

## References:
This repo is built upon [Replica Exchange Stochastic Gradient MCMC](https://github.com/WayneDW/Variance_Reduced_Replica_Exchange_SGMCMC/tree/main)

1. Max Welling, and Yee W. Teh. "[Bayesian Learning via Stochastic Gradient Langevin Dynamics.](https://dl.acm.org/doi/10.5555/3104482.3104568)" ICML 2011.
2. Wei Deng, Qi Feng, Liyao Gao, Faming Liang, and Guang Lin. "[Non-convex Learning via Replica Exchange Stochastic Gradient MCMC.](https://proceedings.mlr.press/v119/deng20b.html)" ICML 2020.
3. Wei Deng, Qi Feng, Georgios P. Karagiannis, Guang Lin, and Faming Liang. "[Accelerating Convergence of Replica Exchange Stochastic Gradient MCMC via Variance Reduction.](https://openreview.net/forum?id=iOnhIy-a-0n)" ICLR 2020.
4. Ruqi Zhang, Chunyuan Li, Jianyi Zhang, Changyou Chen, and Andrew Gordon Wilson. "[Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning.](https://openreview.net/forum?id=rkeS1RVtPS)" ICLR 2020.
5. Haoyang Zheng, Hengrong Du, Qi Feng, Wei Deng, and Guang Lin. "[Constrained Exploration via Reflected Replica Exchange Stochastic Gradient Langevin Dynamics.]()" ICML 2024.
