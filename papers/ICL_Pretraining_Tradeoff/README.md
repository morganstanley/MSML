# How Does the Pretraining Distribution Shape In-Context Learning? A Fundamental Trade-Off

Authors: Waïss Azizian, Ali Hasan

Paper: [arxiv](https://arxiv.org/abs/2510.01163)


## Abstract

The factors driving the performance of in-context learning (ICL) in large language
models (LLMs) remain poorly understood despite ICL's surprising effectiveness, enabling
models to adapt to new tasks from only a handful of examples. To clarify and improve
these capabilities, we characterize how the statistical properties of the pretraining
distribution (e.g., tail behavior, coverage) shape ICL. We develop a theoretical
framework that encompasses generalization and task selection and show how distributional
properties govern sample efficiency, task retrieval, and robustness. To this end, we
generalize existing concentration results to heavy-tailed priors and dependent
sequences, better reflecting the structure of LLM pretraining data. Our framework
reveals a fundamental design trade-off: heavy-tailed pretraining distributions
facilitate robust task selection under distribution shifts but are detrimental to
generalization, especially in low-data regimes. We then empirically evaluate our
predictions by studying how ICL performance varies with the pretraining distribution on
challenging tasks such as stochastic differential equations and stochastic processes
with memory. Together, these findings suggest that controlling key statistical
properties of the pretraining distribution is essential for building ICL-capable and
reliable LLMs.

## Publications

[International Conference on Machine Learning (ICML), 2026](https://openreview.net/forum?id=QiTBmWiH8G)

## Code

Code is adapted from [the original code in this repository](https://github.com/mansheej/icl-task-diversity),
released under the Apache 2.0 license with

> Raventós, A., Paul, M., Chen, F., & Ganguli, S. (2023). Pretraining task diversity and
> the emergence of non-Bayesian in-context learning for regression. Advances in Neural
> Information Processing Systems, 36, 14228-14246.

and extended for this work with new pretraining distributions, task reweighting,
Ornstein-Uhlenbeck and Volterra tasks, and the associated analysis tooling.

## Setup

Python 3.10 is recommended.

```
conda create -n icl -y python=3.10
conda activate icl
```

Install JAX first, with the wheel matching your accelerator (see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)):

```
pip install "jax[cpu]"      # CPU
# pip install "jax[cuda12]" # NVIDIA GPU
# pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Then install the remaining dependencies and the package itself:

```
pip install -r requirements.txt
pip install -e .
```

The experiments in the paper were run on TPU with the exact versions pinned in
`requirements-tpu-lock.txt`; that file is kept for reference and is not required to run
the code.

## Experiments

Experiments are configured with [Hydra](https://hydra.cc). Configuration files live in
`icl/configs`, and a run is selected by its file name (no path, no `.yaml`):

```
python run.py --config-name=generalization_student
```

Any field can be overridden from the command line:

```
python run.py --config-name=generalization_student task.n_tasks=4 model.n_layer=6
```

Hydra sweeps run several configurations in one command:

```
python run.py --multirun --config-name=reweighting_student task.distrib_param=3.0,5.0,10.0
```

Each run writes to `outputs/YYYY-MM-DD_HH-MM-SS/`, and each sweep to
`outputs/multirun/YYYY-MM-DD_HH-MM-SS/<job number>/`. A run directory contains the
resolved configuration (`config.json`), the console log (`run.log`), the structured
metrics (`log.json`), and model checkpoints as `.safetensors`.

The configurations reproducing the linear regression experiments are:

| Config | Experiment |
| ------ | ---------- |
| `generalization_student.yaml` | Generalization, Student-t prior |
| `generalization_gen.yaml` | Generalization, generalized-normal prior |
| `reweighting_student.yaml` | Reweighting, Student-t prior |
| `reweighting_gen.yaml` | Reweighting, generalized-normal prior |
| `variance.yaml` | Reweighting with generalized-normal prior, variance analysis |

For the Ornstein-Uhlenbeck experiments:

| Config | Experiment |
| ------ | ---------- |
| `ou_student.yaml` | Student-t prior |
| `ou_gen.yaml` | Generalized-normal prior |

For the Volterra experiments: `volterra.yaml`.

The remaining configurations in `icl/configs` are development and ablation variants;
`fast*.yaml` are reduced-size configurations useful for debugging.

## Analysis

`analyze.py` reads the outputs of a run and produces the plots and fitted metrics. It is
run from the directory containing `outputs/`:

```
python analyze.py                                    # most recent run
python analyze.py 2025-08-06_12-24-25                # a specific run
python analyze.py --multirun                         # most recent multirun
python analyze.py --multirun --shift-analysis 2025-08-11_11-45-46
python analyze.py --multirun --icl-plots 2025-08-11_11-45-46
```

`python analyze.py --help` lists all analysis modes: `--shift-analysis` (task-shift
power-law fits), `--hyperparam-analysis` (hyperparameter heatmaps), `--weights`
(reweighting-weight diagnostics), `--min-mse-analysis`, `--opt-icl-plots`,
`--icl-plots`. Each analysis also exists as an importable module: `task_shift.py`,
`training_analysis.py`, `weight_analysis.py`, `hyperparam_analysis.py`, `icl_plots.py`,
`opt_icl_plots.py`, `mean_min_best_mse.py`, with log and checkpoint loading in
`loading.py`.

`plot_student_pdfs.py` and `plot_gennormal_pdfs.py` reproduce the prior-density figures;
`plot_ou.py` plots Ornstein-Uhlenbeck sample paths.

## Repository layout

```
icl/            core package
  configs/      Hydra configuration files
  train.py      training loop
  models.py     transformer architectures
  tasks.py      task / pretraining-distribution generators
  evaluate.py   evaluation and analytical baselines
  reweighting.py  task-reweighting strategies
  gpt2.py       GPT-2 style transformer
  optim.py      optimizers and schedules
  utils.py      shared utilities
run.py          experiment entry point
analyze.py      analysis and plotting entry point
loading.py      helpers for reading logs and checkpoints
```

## Citation

```
@inproceedings{azizian2026how,
    author = {Azizian, Wa{\"\i}ss and Hasan, Ali},
    title = {How Does the Pretraining Distribution Shape In-Context Learning? A Fundamental Trade-Off},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2026},
    url = {https://openreview.net/forum?id=QiTBmWiH8G}
}
```

Please also consider citing the work this code builds on:

```
@inproceedings{raventos2023pretraining,
    author = {Raventos, Allan and Paul, Mansheej and Chen, Feng and Ganguli, Surya},
    title = {Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {36},
    pages = {14228--14246},
    year = {2023}
}
```

## License

All source files in this repository, unless explicitly mentioned otherwise, are released
under the Apache 2.0 license, the text of which can be found in the LICENSE file.

## Contact

authors: [waiss.azizian@morganstanley.com](mailto:waiss.azizian@morganstanley.com), [ali.hasan@morganstanley.com](mailto:ali.hasan@morganstanley.com)

Morgan Stanley Machine Learning Research: [msml-qa@morganstanley.com](mailto:msml-qa@morganstanley.com)
