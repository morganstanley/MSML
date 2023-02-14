# Risk Bounds on Aleatoric Uncertainty Recovery
This is the code repository for Zhang et al. (2023). "Risk bounds on aleatoric uncertainty recovery"; _AISTATS 2023, accepted_ 

## To replicate the settings and runs in synthetic data experiments

- Check ./configs/ for config specification
    * simdata.yaml: config for synthetic data generation
    * defaults.yaml, ds1-ds5.yaml: some hyperparameter setup for model training
- To generate synthetic data (with replicas) and put it in the designated format to be used in training; multiple datasets can be generated all at once, with their identifiers separated by comma:
    ```console
    python run_simdata_prep.py --config='configs/simdata.yaml' --ds_strs='ds1,ds2,ds3,ds4,ds5'
    ```
- To train on a specific synthetic dataset using neural network (for ds1, ds2 and ds3 in paper)
    ```console
    python train_sim.py --ds_str='ds1' --cuda=0 --n_replica=10 --train_size=1000 --use_true_mean=0
    ```
- To train on a specific synthetic dataset with functional basis provided (for ds4 and ds5 in paper); here we use --use\_true\_mean=1 for the reason elaborated in the corresponding section in the paper
    ```console
    python train_sim_fb.py --ds_str='ds5' --n_replica=10 --train_size=1000 --use_true_mean=1
    ```
    
NOTE: the exact underlying data and results could vary depending on the random seeds used during the data generation and training process. 
    
## Repo structure
    ```
        ├── README.md
        ├── configs                     ## config files for synthetic data generation and model training
        │   ├── defaults.yaml
        │   ├── ds1.yaml
        │   ├── ds2.yaml
        │   ├── ds3.yaml
        │   ├── ds4.yaml
        │   ├── ds5.yaml
        │   └── simdata.yaml
        ├── helpers                     ## various helper classes
        │   ├── __init__.py
        │   ├── constructor.py
        │   ├── evaluator.py
        │   └── simulator.py
        ├── src                         ## networks used to parametrize mean and variance; estimator wrapper
        │   ├── __init__.py
        │   ├── estimator.py
        │   ├── fbestimator.py
        │   ├── fcresnet.py
        │   ├── losses.py
        │   └── mlp.py
        ├── output_sim                  ## output folder where results for synthetic data experiments would be saved
        │   ├── paper                   ## plots and tables included in the paper
        │   │   ├── fig1.png
        │   │   ├── fig2.png
        │   │   ├── fig3.png
        │   │   ├── fig4.png
        │   │   ├── table1.csv
        │   │   └── table2.csv
        │   └── process_results.ipynb   ## notebook that processed the results coming out of synthetic data experiments
        ├── run_simdata_prep.py         ## script for synthetic data generation and train/val/test split
        ├── train_sim.py                ## train mean and variance based on NNs, for ds1, ds2 and ds3
        ├── train_sim_fb.py             ## train variance using functional basis; for ds4 and ds4
    ```
