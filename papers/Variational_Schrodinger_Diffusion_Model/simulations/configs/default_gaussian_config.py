import ml_collections

def get_gaussian_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed=42 #The utlimate answer of UNIVERSE!
  config.T = 1.0
  config.interval = 100
  config.train_method = 'alternate'
  config.t0 = 0.001
  config.problem_name = 'spiral'
  config.num_itr = 100
  config.eval_itr = 200
  config.forward_net = 'Linear'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.num_epoch = 1
  config.num_stage = 1
  config.train_bs_x = 1000 
  # sampling
  config.samp_bs = 3000
  config.beta_min = 0.1
  config.beta_max = 20
  config.snapshot_freq = 10
  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 1e-3
  config.lr_orthogonal = 3e-4
  config.lr_gamma = 0.99

  model_configs=None

  """ alternative hyperparameters """
  config.train_bs_x_dsm = 200
  config.train_bs_t_dsm = 200
  #config.train_bs_t = 4
  config.num_itr_dsm = 500
  config.DSM_warmup = False 

  return config, model_configs
