# Recurrent Interpolants for Probabilistic Time Series Prediction

Authors: Yu Chen, Marin Biloš, Sarthak Mittal, Wei Deng, Kashif Rasul, Anderson Schneider

Paper: [arxiv](https://arxiv.org/abs/2409.11684)


## Abstract

Sequential models such as recurrent neural networks or transformer-based models became *de facto* tools for multivariate time series forecasting in a probabilistic fashion, with applications to a wide range of datasets, such as finance, biology, medicine, etc. Despite their adeptness in capturing dependencies, assessing prediction uncertainty, and efficiency in training, challenges emerge in modeling high-dimensional complex distributions and cross-feature dependencies. To tackle these issues, recent works delve into generative modeling by employing diffusion or flow-based models. Notably, the integration of stochastic differential equations or probability flow successfully extends these methods to probabilistic time series imputation and forecasting. However, scalability issues necessitate a computational-friendly framework for large-scale generative model-based predictions. This work proposes a novel approach by blending the computational efficiency of recurrent neural networks with the high-quality probabilistic modeling of the diffusion model, which addresses challenges and advances generative models' application in time series forecasting. Our method relies on the foundation of stochastic interpolants and the extension to a broader conditional generation framework with additional control features, offering insights for future developments in this dynamic field.

## Setup

Install `time_match` package locally. This will install all the dependencies.
```
pip install -e .
```

## Experiments

The following are the scripts that run all the experiments.

Synthetic:
```
. scripts/run_synthetic
```

Forecasting:
```
. scripts/run_forecasting
```

AR:
```
. scripts/run_ar
```

You can also run individual experiments using python scripts. An example for synthetic data:
```
python -m time_match.experiment.synthetic --seed 1 --target swissroll-L --model ddpm-linear --net resnet-3 --device 0
```

## Citation

```
@article{chen2024recurrent,
  title={Recurrent Interpolants for Probabilistic Time Series Prediction},
  author={Chen, Yu and Bilo{\v{s}}, Marin and Mittal, Sarthak and Deng, Wei and Rasul, Kashif and Schneider, Anderson},
  journal={arXiv:2409.11684},
  year={2024}
}
```

## Licence

All source files in this repository, unless explicitly mentioned otherwise, are released under the Apache 2.0 license, the text of which can be found in the LICENSE file.

## Contact

Morgan Stanley Machine Learning Research: msml-qa@morganstanley.com
