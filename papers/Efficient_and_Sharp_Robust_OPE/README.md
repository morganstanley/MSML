# Efficient and Sharp Off-policy Evaluation in Robust Markov Decision Processes

Authors: Andrew Bennett, Nathan Kallus, Miruna Oprescu, Wen Sun, Kaiwen Wang


## Abstract

We study evaluating a policy under best- and worst-case perturbations to a Markov decision process (MDP), given transition observations from the original MDP, whether under the same or different policy. This is an important problem when there is the possibility of a shift between historical and future environments, due to \emph{e.g.} unmeasured confounding, distributional shift, or an adversarial environment. We propose a perturbation model that can modify transition kernel densities up to a given multiplicative factor or its reciprocal, which extends the classic marginal sensitivity model (MSM) for single time step decision making to infinite-horizon RL. We characterize the sharp bounds on policy value under this model, that is, the tightest possible bounds given by the transition observations from the original MDP, and we study the estimation of these bounds from such transition observations. We develop an estimator with several appealing guarantees: it is semiparametrically efficient, and remains so even when certain necessary nuisance functions such as worst-case Q-functions are estimated at slow nonparametric rates. It is also asymptotically normal, enabling easy statistical inference using Wald confidence intervals. In addition, when certain nuisances are estimated inconsistently we still estimate valid, albeit possibly not sharp bounds on the policy value. We validate these properties in numerical simulations. The combination of accounting for environment shifts from train to test (robustness), being insensitive to nuisance-function estimation (orthogonality), and accounting for having only finite samples to learn from (inference) together leads to credible and reliable policy evaluation.

## Publications

- Currently available in prepprint form [here](https://arxiv.org/abs/2404.00099)

## Data

The experiment runs on synthetic data, which is generated via the `build_datasets.py` script. See execution details below.

## Code

### Layout

The following is a quick summary of the code structure:
1. *environments* contains the Environments 
2. *policies* contains different policies we experimented with
3. *models* contains nn.Module code for modelling the Q / beta / w functions (along with the critic functions for minimax methods)
4. *learners* contains the learning algorithms (both robust FQI for Q estimation, and sieve-based minimax learning for W estimation)
5. *utils* contains some useful generic utilities

### Setup

Versions of important libraries used are in requirements.txt, which can be installed via `pip install -r requirements.txt`. All experiments were run in python 3.10.


### Execution

In order to run the code and repreoduce experimental results, you need to:
1. run `build_datasets.py` to build the datasets for the N dataset replications
2. run `run_experiments.py` to run the main experiments. This by default uses the config file at `configs/experiment_config.json`, but can be changed using --config command line arg
3. run `get_true_policy_values.py` to compute the true worst-case policy values for each Lambda (needed for creating plots)
4. run `create_results_plots.py` to build the result plot and table of MSE values

Note that some parts of the configuration file might need to be modified for code to run correctly on the target machine. Of important note:
- "num_workers" specifies the number of parallel processes to run for experiments (default is 4, but this can be increased to speed up experiments)
- "devices" lists the devices used in torch (i.e. GPUs and/or CPUs) for running experiments on. Each worker process will be asigned a device from the list of available devices procided in the config file in a round-robin fashion. By default we set devices=[0], meaning that everything can be run on device 0, but this can be modified to run on multiple GPU, or this key can be deleted from the config file to just run everything on CPU.

In addition, the experiment can be run in different ways, e.g. with more replications or different hyperparameters, by modifying the config appropriately.


## Citations

```
@article{bennett2024efficient,
  title={Efficient and Sharp Off-Policy Evaluation in Robust Markov Decision Processes},
  author={Bennett, Andrew and Kallus, Nathan and Oprescu, Miruna and Sun, Wen and Wang, Kaiwen},
  journal={arXiv preprint arXiv:2404.00099},
  year={2024}
}
```


## License

All source files in this repository, unless explicitly mentioned
otherwise, are released under the Apache 2.0 license, the text of
which can be found in the LICENSE file.


## Contact

author: [andrew.bennett@morganstanley.com](mailto:andrew.bennett@morganstanley.com)

Morgan Stanley Machine Learning Research: [msml-qa@morganstanley.com](mailto:msml-qa@morganstanley.com)
