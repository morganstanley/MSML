import ml_collections

def get_gmm_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42
  config.T = 1.0
  config.interval = 100
  config.train_method = 'joint'
  config.t0 = 0
  config.problem_name = 'gmm'
  config.num_itr = 2000
  config.eval_itr = 200
  config.forward_net = 'toy'
  config.backward_net = 'toy'

  # sampling
  config.samp_bs = 1000
  config.sigma_min = 0.01
  config.sigma_max = 5

  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 1e-4
  config.lr_gamma = 0.9

  model_configs=None
  return config, model_configs

