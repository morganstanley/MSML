import ml_collections


def get_cifar10_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42
  config.train_bs_x_dsm = 54
  config.train_bs_t_dsm = 3
  config.train_bs_x = 2
  config.train_bs_t = 32
  config.num_stage = 20
  config.num_epoch = 5
  config.num_itr = 300
  config.T = 1.0
  config.interval = 201
  config.train_method = 'alternate'
  config.t0 = 1e-4
  config.lr_gamma = 0.99
  config.FID_freq = 2
  config.snapshot_freq = 1
  config.ckpt_freq = 2
  config.num_FID_sample = 10000
  config.problem_name = 'cifar10'
  # sampling
  config.snr = 0.16
  config.samp_bs = 500
  config.num_itr_dsm = 200000
  config.DSM_warmup = False

  config.sigma_min = 0.01
  config.sigma_max = 50
  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 1e-5
  config.grad_clip = 2.
  config.scale_by_g = True

  model_configs={'ncsnpp':get_NCSNpp_config(), 'Unet':get_Unet_config()}
  return config, model_configs


def get_Unet_config():
  config = ml_collections.ConfigDict()
  config.name = 'Unet'
  config.attention_resolutions = '16,8'
  # config.attention_layers=[1,2]
  config.in_channels = 3
  config.out_channel = 3
  config.num_head = 4
  config.num_res_blocks = 2 # 2
  config.num_channels = 128 # 128
  config.dropout = 0.1
  config.channel_mult = (1, 2, 2, 2)
  config.image_size = 32
  return config


def get_NCSNpp_config():
    config = get_resolution32_default_configs()
    # training
    training = config.training
    training.sde = 'vesde'
    training.continuous = True
    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = True  # To match Lou setting.
    model.convert_t_to_std = True  # To match Lou setting.
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128 # 128
    model.ch_mult = (1, 2, 2, 2) #
    model.num_res_blocks = 8 #  8 in Lou 2022
    model.attn_resolutions = (16,) #
    model.resamp_with_conv = True
    model.conditional = True # ?
    model.fir = False # default True
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
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.dropout = 0.1
  return config


