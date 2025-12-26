# Reproducing our experiments

In the following, we describe which scripts and notebooks to run in which order to reproduce the different figures and tables from our paper.

Note that you may have to adjust the directories in the individual config files (to point at the correct dataset folders, result folders etc.).  
You have to manually create these directories, they are not automatically created by the program itself.  
You will also need to adjust the slurm configuration parameters at the top of each file to match your cluster configuration (partition names etc.).

## Figures

### Optimal choice of number of subsequences under composition - Figs. 2, 5-7
```
seml dp_timeseries_eval_pld_monotonicity_composed add seml/configs/dp_only/eval_pld_monotonicity_composed/wor_top_wr_bottom.yaml start
```
Then run `plotting/dp_only/eval_pld_monotonicity_composed/wor_top_wr_bottom.ipynb`.

### Deterministic vs WOR top-level - Figs. 3, 8
```
seml dp_timeseries_eval_pld_deterministic_vs_random_top_level add seml/configs/dp_only/eval_pld_deterministic_vs_random_top_level/wor_top_wr_bottom.yaml start
```
Then run `plotting/dp_only/eval_pld_deterministic_vs_random_top_level/wor_top_wr_bottom.ipynb`.

### Amplification by label perturbation - Figs. 4, 9
```
seml dp_timeseries_eval_pld_label_noise add seml/configs/dp_only/eval_pld_label_noise/wor_top_wr_bottom.yaml start
```
Then run `plotting/dp_only/eval_pld_label_noise/wor_top_wr_bottom.ipynb`.

### Epoch privacy vs length - Fig. 10

```
seml dp_timeseries_eval_pld_deterministic_epoch_length add seml/configs/dp_only/eval_pld_deterministic_epoch_length/deterministic_top_wr_bottom.yaml start
```
Then run `plotting/dp_only/eval_pld_deterministic_epoch_length/deterministic_top_wr_bottom.ipynb`.

### Effect of Subsampling Parameters on Non-Private Utility - Figs. 11, 12
```
seml dp_timeseries_standard_train_standard_eval add seml/configs/traffic/standard_train_standard_eval_batch_size.yaml start
seml dp_timeseries_standard_train_standard_eval add seml/configs/traffic/standard_train_standard_eval_num_instances.yaml start
```
Then run `plotting/non_dp_utility/standard_train_batch_size/traffic.ipynb` and `plotting/non_dp_utility/standard_train_num_instances/traffic.ipynb`.

### Comparison to Standard DP-SGD - Fig. 13
Run `plotting/dp_only/eval_pld_bilevel_vs_blackbox/wor_top_wr_bottom.ipynb`.

## Tables

To apply the traditional forecasting baselines:
```
seml dp_timeseries_traditional_baselines_standard_eval add seml/configs/baselines/traditional_baselines_standard_eval.yaml start
```

### Event-level privacy - Tables 1-4
```
seml dp_timeseries_standard_train_standard_eval add seml/configs/traffic/standard_train_standard_eval.yaml start
seml dp_timeseries_standard_train_standard_eval add seml/configs/electricity/standard_train_standard_eval.yaml start
seml dp_timeseries_standard_train_standard_eval add seml/configs/solar_10_minutes/standard_train_standard_eval.yaml start

seml dp_timeseries_dp_train_standard_eval add seml/configs/traffic/dp_train_standard_eval.yaml start
seml dp_timeseries_dp_train_standard_eval add seml/configs/electricity/dp_train_standard_eval.yaml start
seml dp_timeseries_dp_train_standard_eval add seml/configs/solar_10_minutes/dp_train_standard_eval.yaml start
```
Then run the notebooks in `plotting/final_tables/individual`.

### w-event and w-user-level privacy - Tables 5-7
```
seml dp_timeseries_dp_train_standard_eval_user_level add seml/configs/traffic/dp_train_standard_eval_user_level.yaml start
seml dp_timeseries_dp_train_standard_eval_user_level add seml/configs/electricity/dp_train_standard_eval_user_level.yaml start
seml dp_timeseries_dp_train_standard_eval_user_level add seml/configs/solar_10_minutes/dp_train_standard_eval_user_level.yaml start
```
Then run the notebooks in `plotting/final_tables/user`.

### Amplification by label perturbation - Tables 8-10
```
seml dp_timeseries_dp_train_standard_eval_label_noise add seml/configs/traffic/dp_train_standard_eval_label_noise.yaml start
seml dp_timeseries_dp_train_standard_eval_label_noise add seml/configs/electricity/dp_train_standard_eval_label_noise.yaml start
seml dp_timeseries_dp_train_standard_eval_label_noise add seml/configs/solar_10_minutes/dp_train_standard_eval_label_noise.yaml start
```
Then run the notebooks in `plotting/final_tables/label_noise`.

### Inference privacy - Tables 11-13
```
seml dp_timeseries_traditional_baselines_dp_eval add seml/configs/baselines/traditional_baselines_dp_eval.yaml start

seml dp_timeseries_standard_train_dp_eval add seml/configs/traffic/standard_train_dp_eval.yaml start
seml dp_timeseries_standard_train_dp_eval add seml/configs/electricity/standard_train_dp_eval.yaml start
seml dp_timeseries_standard_train_dp_eval add seml/configs/solar_10_minutes/standard_train_dp_eval.yaml start
```
Then run the notebooks in `plotting/final_tables/inference_privacy`.