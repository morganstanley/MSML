import ml_collections
import torch


def get_mnist_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42
  config.train_bs_x_dsm = 16
  config.train_bs_t_dsm = 8
  config.train_bs_x = 18
  config.train_bs_t = 4
  config.num_stage = 10
  config.num_epoch = 10
  config.num_itr = 400
  config.T = 1.0
  config.train_method = 'alternate'
  config.t0 = 1e-3
  config.lr_gamma = 0.99
  config.FID_freq = 2
  config.snapshot_freq = 2
  config.ckpt_freq = 2
  config.num_FID_sample = 10000
  config.problem_name = 'mnist'
  config.num_itr_dsm = 10000
  config.DSM_warmup = True

  # sampling
  config.snr = 0.16
  config.samp_bs = 1000

  config.sigma_min = 0.01
  config.sigma_max = 50
  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 5e-4
  config.grad_clip = 1.

  model_configs={'Unet':get_Unet_config()}
  return config, model_configs

def get_Unet_config():
  config = ml_collections.ConfigDict()
  config.name = 'Unet'
  config.attention_resolutions='16,8'
  config.in_channels = 1
  config.out_channel = 1
  config.num_head = 2
  config.num_res_blocks = 2
  config.num_channels = 32
  config.dropout = 0.0
  config.channel_mult = (1, 1, 2, 2)
  config.image_size = 32 # since we have padding=2
  return config
