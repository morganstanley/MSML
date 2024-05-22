import ml_collections
import torch


def get_sinusoid_large_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42
  config.train_bs_x_dsm = 128
  config.train_bs_t_dsm = 1
  config.train_bs_x = 12
  config.train_bs_t = 12
  config.num_stage = 10
  config.num_epoch = 10
  config.num_itr = 400
  config.T = 1.0
  config.train_method = 'alternate'
  config.t0 = 1e-3
  config.lr_gamma = 0.99
  config.FID_freq = 2
  config.snapshot_freq = 1
  config.ckpt_freq = 1
  config.num_FID_sample = 2000
  config.problem_name = 'sinusoid'
  config.num_itr_dsm = 10000
  # config.DSM_warmup = True
  config.log_tb = True

  # sampling
  config.snr = 0.08
  config.samp_bs = 1000

  config.interval = 100
  config.sigma_min = 0.001
  config.sigma_max = 10.0
  config.beta_min = 0.0001
  config.beta_max = 20

  # optimization
  # config.optim = optim = ml_collections.ConfigDict()
  config.optimizer = 'AdamW'
  config.lr = 5e-4
  config.grad_clip = 1.

  config.input_size = (40, 48)  # (K,L)
  config.permute_batch = True
  model_configs={
      'toyv2':get_Toyv2_config(config.interval, config.input_size),
      'toyv3':get_Toyv3_config(config.interval, config.input_size),
      'Unetv2':get_Unetv2_config(config.input_size),
      'Unetv2_large':get_Unetv2_large_config(config.input_size),
      'Transformerv0':get_Transformerv0_config(config.interval, config.input_size),
      'Transformerv1':get_Transformerv1_config(),
      'Transformerv2':get_Transformerv2_config(),
      'Transformerv3':get_Transformerv3_config(),
      'Transformerv4':get_Transformerv4_config(),
      'Transformerv5':get_Transformerv5_config(),
  }
  return config, model_configs


def get_Toyv2_config(interval, input_size):  # Default
  config = ml_collections.ConfigDict()
  config.name = 'toyv2'
  config.interval = interval
  config.channels = [64, 64, 64, 64]  # [32, 64, 128, 256]
  config.embed_dim = 256
  config.input_size = input_size
  return config


def get_Toyv3_config(interval, input_size):  # Default
  config = ml_collections.ConfigDict()
  config.name = 'toyv3'
  config.interval = interval
  config.channels = [128, 128, 128,]
  config.embed_dim = 256
  config.input_size = input_size
  return config


def get_Unetv2_config(input_size=None):  # Default
  config = ml_collections.ConfigDict()
  config.name = 'Unetv2'
  config.attention_layers=[1,2]
  config.in_channels = 1
  config.out_channel = 1
  config.num_head = 2
  config.num_res_blocks = 2
  config.num_channels = 32
  config.num_norm_groups = 32  # num_channels is divisible by num_norm_groups.
  config.dropout = 0.0
  config.channel_mult = (1, 1, 1)  # (1, 1, 2, 2)
  config.input_size = input_size
  return config


def get_Unetv2_large_config(input_size=None):  # Default
  config = ml_collections.ConfigDict()
  config.name = 'Unetv2'
  config.attention_layers=[1,2]
  config.in_channels = 1
  config.out_channel = 1
  config.num_head = 4
  config.num_res_blocks = 2
  config.num_channels = 64
  config.num_norm_groups = 32  # num_channels is divisible by num_norm_groups.
  config.dropout = 0.1
  config.channel_mult = (1, 1, 1,)  # (1, 1, 2, 2)
  config.input_size = input_size
  return config


# def get_Unetv2_config(input_size=None):  # Small
#   config = ml_collections.ConfigDict()
#   config.name = 'Unetv2'
#   config.attention_layers=[1,2]
#   config.in_channels = 1
#   config.out_channel = 1
#   config.num_head = 2
#   config.num_res_blocks = 2
#   config.num_channels = 16
#   config.num_norm_groups = 16  # num_channels is divisible by num_norm_groups.
#   config.dropout = 0.0
#   config.channel_mult = (1, 1, 1)
#   config.input_size = input_size
#   return config


def get_Transformerv0_config(interval, input_size):  # Default
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv0'
  config.layers = 2
  config.nheads = 8
  config.diff_channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = (8, 50)

  # For toy model part.
  config.interval = interval
  config.channels = [64, 64, 64, 64]  # [32, 64, 128, 256]
  # config.embed_dim = 256
  config.input_size = input_size
  return config


def get_Transformerv1_config():  # Default
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv1'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = (8, 50)
  config.output_layer = 'conv1d'  # conv1d as default.
  return config


# def get_Transformerv1_config():  # Small
#   config = ml_collections.ConfigDict()
#   config.name = 'Transformerv1'
#   config.layers = 3
#   config.nheads = 8
#   config.channels = 48
#   config.diffusion_embedding_dim = 64
#   config.timeemb = 64
#   config.featureemb = 16
#   config.input_size = (8, 50)
#   config.output_layer = 'conv2d'  # conv1d as default.
#   return config


def get_Transformerv2_config():
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv2'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = (8, 50)
  config.output_layer = 'conv1d'  # conv1d as default.
  return config


def get_Transformerv3_config():
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv3'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = (8, 50)
  config.output_layer = 'conv2d'  # conv1d as default.
  return config

def get_Transformerv4_config():
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv4'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = (8, 50)
  config.output_layer = 'conv1d'  # conv1d as default.
  return config

def get_Transformerv5_config():
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv5'
  config.layers = 4
  config.nheads = 8
  config.channels = 56
  config.diffusion_embedding_dim = 128
  config.timeemb = 64
  config.featureemb = 16
  config.input_size = (8, 50)
  config.output_layer = 'conv1d'  # conv1d as default.
  return config


