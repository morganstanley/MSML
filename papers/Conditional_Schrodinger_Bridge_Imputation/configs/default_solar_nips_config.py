import ml_collections
import torch


def get_solar_nips_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 0
  config.train_bs_x_dsm = 64
  config.train_bs_t_dsm = 1
  config.train_bs_x = 12
  config.train_bs_t = 12
  config.num_stage = 10
  config.num_epoch = 10
  config.num_itr = 400
  config.T = 1.0
  config.train_method = 'dsm_imputation_v2'
  config.t0 = 1e-3
  config.FID_freq = 1
  config.snapshot_freq = 1
  config.ckpt_freq = 1
  config.num_FID_sample = 2000
  config.problem_name = 'solar_nips'
  config.num_itr_dsm = 10000
  # config.DSM_warmup = True
  config.log_tb = True

  # sampling
  config.snr = 0.08
  config.samp_bs = 1000

  config.interval = 50
  config.sigma_min = 0.001
  config.sigma_max = 20.0
  config.beta_min = 0.001
  config.beta_max = 20

  # optimization
  # config.optim = optim = ml_collections.ConfigDict()
  config.optimizer = 'AdamW'
  config.lr = 5e-4
  # config.l2_norm = 1e-6
  config.grad_clip = 1.
  config.lr_gamma = 0.99
  config.ema_decay = 0.99

  config.input_size = (137, 80+24)  # (K,L)
  # config.target_dim_range = (72, 108)
  # config.input_size = (16, 100+24)  # (K,L)
  config.permute_batch = True
  config.imputation_eval = True
  config.output_layer = 'conv1d'  # 'conv1d_silu'

  model_configs={
      'Unetv2':get_Unetv2_config(config.input_size),
      'Transformerv1':get_Transformerv1_config(config.input_size, config.output_layer),
      'Transformerv2':get_Transformerv2_config(config.input_size, config.output_layer),
      'Transformerv3':get_Transformerv3_config(config.input_size, config.output_layer),
      'Transformerv4':get_Transformerv4_config(config.input_size, config.output_layer),
      'Transformerv5':get_Transformerv5_config(config.input_size, config.output_layer),
  }
  return config, model_configs


def get_Unetv2_config(input_size=None):
  config = ml_collections.ConfigDict()
  config.name = 'Unetv2'
  # config.attention_resolutions='16,8'
  config.attention_layers=[1,2]
  config.in_channels = 1
  config.out_channel = 1
  config.num_head = 2
  config.num_res_blocks = 2
  config.num_channels = 128  # 128 <-- Due to GPU memory limitation.
  config.num_norm_groups = 32  # num_channels is divisible by num_norm_groups.
  config.dropout = 0.0
  config.channel_mult = (1,1,1) # (1, 1, 1)  # (1, 1, 2, 2)  # 128 <-- Due to GPU memory limitation.
  config.input_size = input_size # since we have padding=2
  return config


def get_Transformerv1_config(input_size=None, output_layer='conv1d'):  # Default
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv1'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = input_size
  config.output_layer = output_layer  # conv1d as default.
  return config


# def get_Transformerv1_config(input_size=None, output_layer='conv1d'):  # Small
#   config = ml_collections.ConfigDict()
#   config.name = 'Transformerv1'
#   config.layers = 3
#   config.nheads = 6
#   config.channels = 48
#   config.diffusion_embedding_dim = 128
#   config.timeemb = 64
#   config.featureemb = 16
#   config.input_size = input_size
#   config.output_layer = output_layer  # conv1d as default.
#   return config


def get_Transformerv2_config(input_size=None, output_layer='conv1d'):  # Default model.
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv2'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128  # 128 <-- Due to GPU memory limitation.
  config.featureemb = 32
  config.input_size = input_size
  config.output_layer = output_layer  # conv1d as default.
  return config


# def get_Transformerv2_config(input_size=None, output_layer='conv1d'):  # Large model, SOTA
#   config = ml_collections.ConfigDict()
#   config.name = 'Transformerv2'
#   config.layers = 6
#   config.nheads = 8
#   config.channels = 96
#   config.diffusion_embedding_dim = 64
#   config.timeemb = 64
#   config.featureemb = 32
#   config.input_size = input_size
#   config.output_layer = output_layer  # conv1d as default.
#   return config


# def get_Transformerv2_config(input_size=None, output_layer='conv1d'):  # Larger model
#   config = ml_collections.ConfigDict()
#   config.name = 'Transformerv2'
#   config.layers = 6
#   config.nheads = 8
#   config.channels = 96
#   config.diffusion_embedding_dim = 128
#   config.timeemb = 64
#   config.featureemb = 32
#   config.input_size = input_size
#   config.output_layer = output_layer  # conv1d as default.
#   return config


# def get_Transformerv2_config(input_size=None, output_layer='conv1d'):  # Small model.
#   config = ml_collections.ConfigDict()
#   config.name = 'Transformerv2'
#   config.layers = 2
#   config.nheads = 4
#   config.channels = 32
#   config.diffusion_embedding_dim = 48
#   config.timeemb = 48
#   config.featureemb = 16
#   config.input_size = input_size
#   config.output_layer = output_layer  # conv1d as default.
#   return config


def get_Transformerv3_config(input_size=None, output_layer='conv1d'):  # Default mode.
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv3'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = input_size
  config.output_layer = output_layer  # conv1d as default.
  return config

def get_Transformerv4_config(input_size=None, output_layer='conv1d'):  # Default model.
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv4'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = input_size
  config.output_layer = output_layer  # conv1d as default.
  return config

def get_Transformerv5_config(input_size=None, output_layer='conv1d'):  # Default model.
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv5'
  config.layers = 4
  config.nheads = 8
  config.channels = 56
  config.diffusion_embedding_dim = 128
  config.timeemb = 64
  config.featureemb = 16
  config.input_size = input_size
  config.output_layer = output_layer  # conv1d as default.
  return config


