# Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion

Marin Biloš, Kashif Rasul, Anderson Schneider, Yuriy Nevmyvaka, Stephan Günnemann

International Conference on Machine Learning (ICML), 2023

Abstract: Temporal data such as time series can be viewed as discretized measurements of the underlying function. To build a generative model for such data we have to model the stochastic process that governs it. We propose a solution by defining the denoising diffusion model in the function space which also allows us to naturally handle irregularly-sampled observations. The forward process gradually adds noise to functions, preserving their continuity, while the learned reverse process removes the noise and returns functions as new samples. To this end, we define suitable noise sources and introduce novel denoising and score-matching models. We show how our method can be used for multivariate probabilistic forecasting and imputation, and how our model can be interpreted as a neural process.

## Simple example

You can find a runnable example of DSPD-GP model in a [self-contained notebook](example.ipynb). For full experiments follow the instructions below.

## Installation

Run to install local package `tsdiff` and other required packages:
```sh
pip install -e .
```

Run tests:
```sh
pytest
```

## Probabilistic modeling

Generate synthetic data:
```sh
python -m tsdiff.data.generate
```

Run example:
```sh
python -m tsdiff.synthetic.train --seed 1 --dataset lorenz --diffusion GaussianDiffusion --model rnn
```
Other options (see [file](tsdiff/synthetic/train.py)):
```
python -m tsdiff.synthetic.train
--dataset [cir|lorenz|ou|predator_prey|sine|sink]
--diffusion [GaussianDiffusion|OUDiffusion|GPDiffusion|ContinuousGaussianDiffusion|ContinuousOUDiffusion|ContinuousGPDiffusion]
--model [feedforward|rnn|transformer]
--gp_sigma [float]
--ou_theta [float]
```

See the files [experiment.py](tsdiff/synthetic/experiment.py) and [experiment.yaml](tsdiff/synthetic/experiment.yaml) for replicating the experiments, as well as [discriminator_experiment.py](tsdiff/synthetic/discriminator_experiment.py) and [discriminator_experiment.yaml](tsdiff/synthetic/discriminator_experiment.yaml) for discriminator experiment.

## Forecasting

Example:
```sh
python -m tsdiff.forecasting.train --seed 1 --dataset electricity_nips --network timegrad_rnn --noise ou --epochs 100
```
Other options can be found in [train.py](tsdiff/forecasting/train.py). See [experiment.py](tsdiff/forecasting/experiment.py) and [experiment.yaml](tsdiff/forecasting/experiment.yaml) for reproducing the experiments.

## Neural process

Example:
```sh
python -m tsdiff.neural_process.train
```
Use [experiment.py](tsdiff/neural_process/experiment.py) and [experiment.yaml](tsdiff/neural_process/experiment.yaml) to reproduce the experiments.


## Imputation experiment

The setup and the code is very similar to the official implementation of CSDI [[arxiv](https://arxiv.org/abs/2107.03502), [github](https://github.com/ermongroup/CSDI)].

In our version, we keep the actual times of the observations, instead of rounding them up to the nearest hour.

Download data:
```sh
python -m tsdiff.csdi.download physio
```

Running the code:
```sh
# original paper
python -m tsdiff.csdi.exe_physio

# our paper
python -m tsdiff.csdi.exe_physio --gp_noise
```
Other parameters include `--is_unconditional`, `--nsample` (e.g., 100), `--testmissingratio` (e.g., 0.5).

Replicating the experiments: `seml tsdiff_csdi add tsdiff/csdi/experiment.yaml start`. File [experiment.yaml](tsdiff/csdi/experiment.yaml) contains the hyperparameter configuration. File [experiment.py](tsdiff/csdi/experiment.py) runs the experiments.


## Citation

```
@inproceedings{bilos2022diffusion,
  title={Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion},
  author={Bilo{\v{s}}, Marin and Rasul, Kashif and Schneider, Anderson and Nevmyvaka, Yuriy and G{\"u}nnemann, Stephan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023},
}
```
