import ml_collections

def get_spiral_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed=42 #The utlimate answer of UNIVERSE!
  config.T = 1.0
  config.interval = 100
  config.train_method = 'alternate'
  config.t0 = 0.001
  config.problem_name = 'spiral'
  config.num_itr = 25
  config.eval_itr = 200
  config.forward_net = 'Linear'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.num_epoch = 1
  config.num_stage = 1
  config.train_bs_x = 1000 
  # sampling
  config.samp_bs = 20000
  config.samp_bs_plot = 3000
  config.beta_min = 1e-2
  config.beta_max = 5
  config.snapshot_freq = 1
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr_f = 3e-2
  config.lr_b = 1e-3
  config.damp_ratio = 0.7
  config.lr_gamma_f = 0.99
  config.lr_gamma_b = 0.99

  model_configs=None

  """ alternative hyperparameters """
  config.train_bs_x_dsm = 200
  config.train_bs_t_dsm = 200
  config.num_itr_dsm = 100
  config.DSM_warmup = False 

  return config, model_configs
