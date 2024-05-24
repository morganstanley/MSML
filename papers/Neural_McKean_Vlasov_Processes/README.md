# Neural McKean-Vlasov Processes: Distributional Dependence in Diffusion Processes

Authors: Haoming Yang, Ali Hasan, Yuting Ng, and Vahid Tarokh 


## Abstract

McKean-Vlasov stochastic differential equations (MV-SDEs) provide a mathematical description of the behavior of an infinite number of interacting particles by imposing a dependence on the particle density. As such, we study the influence of explicitly including distributional information in the parameterization of the SDE. We propose a series of semi-parametric methods for representing MV-SDEs, and corresponding estimators for inferring parameters from data based on the properties of the MV-SDE. We analyze the characteristics of the different architectures and estimators, and consider their applicability in relevant machine learning problems. We empirically compare the performance of the different architectures and estimators on real and synthetic datasets for time series and probabilistic modeling. The results suggest that explicitly including distributional dependence in the parameterization of the SDE is effective in modeling temporal data with interaction under an exchangeability assumption while maintaining strong performance for standard Itô-SDEs due to the richer class of probability flows associated with MV-SDEs.


## Publications

[International Conference on Artificial Intelligience and Statistics (AISTATS), 2024 (Oral)](https://proceedings.mlr.press/v238/yang24a/yang24a.pdf)

## Code

Code is copied from [the original code in this repository.](https://github.com/imkeithyang/Neural-McKean-Vlasov-Processes)

## Citations

```
@inproceedings{neural_mckean_vlasov,
  title={Neural McKean-Vlasov Processes: Distributional Dependence in Diffusion Processes},
  author={Yang, Haoming and Hasan, Ali and Ng, Yuting and Tarokh, Vahid},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={262--270},
  year={2024},
  organization={PMLR}
}
```


## License

Apache 2.0

## Contact

author: [ali.hasan@morganstanley.com](mailto:ali.hasan@morganstanley.com)

Morgan Stanley Machine Learning Research: [msml-qa@morganstanley.com](mailto:msml-qa@morganstanley.com)
