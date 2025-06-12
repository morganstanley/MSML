import ml_collections
import torch


def get_celebA64_default_configs():
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
  config.problem_name = 'celebA64'
  config.num_itr_dsm = 100000
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
  config.lr = 2e-4
  config.grad_clip = 1.
  
  model_configs={'ncsnpp':get_NCSNpp_config(), 'Unet':get_Unet_config()}
  return config, model_configs

def get_Unet_config():
  config = ml_collections.ConfigDict()
  config.name = 'Unet'
  config.attention_resolutions='16,8'
  config.in_channels = 3
  config.out_channel = 3
  config.num_head = 4
  config.num_res_blocks = 4
  config.num_channels = 64
  config.dropout = 0.1
  config.channel_mult = (1, 2, 3, 4)
  config.image_size = 64
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
    model.nf = 128 #
    model.ch_mult = (1, 2, 2, 2) #
    model.num_res_blocks = 4 #
    model.attn_resolutions = (16,) #
    model.resamp_with_conv = True
    model.conditional = True # ?
    model.fir = True #
    model.fir_kernel = [1, 3, 3, 1] #
    model.skip_rescale = True #
    model.resblock_type = 'biggan' #
    model.progressive = 'none' #
    model.progressive_input = 'residual' #
    model.progressive_combine = 'sum' #
    model.attention_type = 'ddpm'#
    model.init_scale = 0.0#
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
  data.image_size = 64
  data.centered = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.dropout = 0.1 #
  return config
