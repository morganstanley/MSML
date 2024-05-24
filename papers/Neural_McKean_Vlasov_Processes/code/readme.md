### Neural Architectures for McKean-Vlasov Processes

The supplementary material contains the code and the yaml files to replicate the results in the submitted paper. The main implementation of our proposed structure is in MeanFieldMLP.py. The parameter estimation for both real and synthetic dataset, as well as the generative modeling experiments can be run using the command:

`python train.py -f yaml_filepath -d device -e experiment_folder`

We also provide our implementation of deepAR in deepAR.py, which serves as benchmarks in real datasets. The deepARs results can be replicated through the command:

`python train_deepAR.py -f yaml_filepath -d device -e experiment_folder`

The generative experiments using the linear Fokker Planck Equation can be replicated through the command:

`python mean_train_fk.py -f yaml_filepath -d device -e experiment_folder`

Finally, the `analysis_ts.py`/`generative_analysis.py` is a script to analyze our result on Synthetic and real TS data (analysis.py) and generative experiments (generative_analysis.py).

The experiments are structured as follows:
* MV-SDE
  * config (All yaml files)
    * add_noise (Synthetic Dataset)
    * OU_jump (OU process with jumps)
    * EEG_by_electrode/EEG_by_electrode_a (non-alcoholics/alcoholics EEG experiments for MLP and MeanFieldMLP)
      * subject1-5
    * EEG_by_deepAR/EEG_by_deepAR_a (non-alcoholics/alcoholics EEG experiments for deepAR with different backbones)
      * subjecta1-5      
    * Chemotaxi (chemotaxi experiments for MLP and MeanFieldMLP)     
    * Chemotaxi_deepAR (chemotaxi experiments for deepAR)     
    * generative_120_10_bridge_eightgauss (generative synthetic eight Gaussian experiments with 120 for 15particles * 8 gaussian, and 10 bridges)
      * 2,10,30,50,100d      
    * generative_100_30_bridge_realGen1_tanh (real data generative experiments with 100 particles and 30 bridges, tanh activation).     
      * power, gas, cortex, hepmass, miniboone     
    * generative_realGen_others (real data generative experiments with other baselines includeing MAF, WGAN, VAE and Score-Based SDE). 
      * score_based_sde includes main.py     
    * appendix_* (experiments in appendix)     


Real Data source (and preprocessing pipeline):         
The two real TS data can be obtained at the following link:        
https://archive.ics.uci.edu/ml/datasets/eeg+database     
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7DF0AT        

Power, Gas, Miniboone, Hepmass preprocessing pipelines and data are obtained from:       
https://github.com/dargilboa/mdma/tree/8ed4afc731936f695d6e320d804e5157e8eb8a71/experiments/UCI        

Cortex data is obtained from:      
https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression     

Our GLOW implementation is adapted from https://github.com/ikostrikov/pytorch-flows.       

MAF implementation is adapted from https://github.com/ikostrikov/pytorch-flows.        
WGAN implementation is adapted from https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan_gp.       
VAE implementation is adapted from https://github.com/AntixK/PyTorch-VAE/tree/master.        
Score-Based SDE implementation is adapted from the official tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht.         