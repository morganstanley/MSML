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
  config.snapshot_freq = 1
  config.ckpt_freq = 1
  config.num_FID_sample = 10000
  config.problem_name = 'mnist'
  config.num_itr_dsm = 10000
  config.DSM_warmup = True
  config.log_tb = True

  # sampling
  config.snr = 0.16
  config.samp_bs = 1000

  config.sigma_min = 0.01
  config.sigma_max = 2
  config.beta_min = 0.0001
  config.beta_max = 0.5
  # optimization
  # config.optim = optim = ml_collections.ConfigDict()
  config.optimizer = 'AdamW'
  config.lr = 5e-4
  config.grad_clip = 1.

  model_configs={
      'toyv2':get_Toyv2_config(),
      'Unet':get_Unet_config(),
      'Unetv2':get_Unetv2_config(),
      'ncsnpp':get_NCSNpp_config(),
  }
  return config, model_configs

def get_Unet_config():
  config = ml_collections.ConfigDict()
  config.name = 'Unet'
  config.attention_resolutions='16,8'
  config.in_channels = 1
  config.out_channel = 1
  config.num_head = 2
  config.num_res_blocks = 2
  config.num_channels = 32  # 32
  config.num_norm_groups = 32  # 32
  config.dropout = 0.0
  config.channel_mult = (1, 1, 2, 2)
  config.image_size = 32 # since we have padding=2
  return config

def get_Toyv2_config():
  config = ml_collections.ConfigDict()
  config.name = 'toyv2'
  config.interval = 100
  config.channels = [64, 128, 256, 256]  # [32, 64, 128, 256]
  config.embed_dim = 256
  config.input_size = (32, 32)
  return config


def get_Unetv2_config():
  config = ml_collections.ConfigDict()
  config.name = 'Unetv2'
  # config.attention_resolutions='16,8'
  config.attention_layers=[1,2]
  config.in_channels = 1
  config.out_channel = 1
  config.num_head = 2
  config.num_res_blocks = 2
  config.num_channels = 32  # 32
  config.num_norm_groups = 32  # 32
  config.dropout = 0.0
  config.channel_mult = (1, 1, 2, 2)
  config.input_size = (32, 32)
  return config


def get_NCSNpp_config():
    config = get_resolution32_default_configs()
    # training
    training = config.training
    training.sde = 'vesde'
    training.continuous = False
    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 64 # 128
    model.ch_mult = (1, 2, 2, 2) #
    model.num_res_blocks = 4 #
    model.attn_resolutions = (16,) #
    model.resamp_with_conv = True
    model.conditional = True # ?
    model.fir = False #
    model.fir_kernel = [1, 3, 3, 1] #
    model.skip_rescale = True #
    model.resblock_type = 'biggan' # biggan (original), ddpm
    model.progressive = 'none' #
    model.progressive_input = 'residual' #
    model.progressive_combine = 'sum' #
    model.attention_type = 'ddpm'#
    model.init_scale = 0.0 #
    if training.continuous:
      model.fourier_scale = 16
      training.continuous = True
      model.embedding_type = 'fourier'
    else:
      model.embedding_type = 'positional' #
    model.conv_size = 3 #?
    return config

def get_resolution32_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()

  # data
  config.data = data = ml_collections.ConfigDict()
  data.image_size = 32 #
  data.centered = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.dropout = 0.1 #
  return config


