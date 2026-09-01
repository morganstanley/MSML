import itertools
import json
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

LOG2 = math.log(2)


@dataclass
class ScddSchedule:
  """SCDD forward process schedule at time t.

  The marginal q(x_t | x_0) is a mixture of clean tokens, uniform noise,
  and masked tokens, parameterized by rho_t (clean fraction) and gamma_t
  (total non-absorbed mass).
  """
  clean_mass: Tensor
  uniform_mass: Tensor
  absorbed_mass: Tensor
  gamma_t: Tensor
  rho_t: Tensor


def _sample_categorical(categorical_probs):
  categorical_probs = categorical_probs.to(torch.float64)
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

def _3d_multinomial_sample(probs):

  B, S, V = probs.shape
  probs_2d = probs.view(-1, V)
  samples = torch.multinomial(probs_2d, 1)
  return samples.view(B, S)

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class ScddDenoisingStep(torch.nn.Module):
  """Wraps _scdd_update as an nn.Module for torch.compile."""
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x, t, dt):
    return self.model._scdd_update(x, t, dt)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs', 'scdd'}
    if self.subs_masking:
      assert self.parameterization in {'d3pm', 'scdd'}

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def on_before_optimizer_step(self, optimizer):
    # Compute gradient norm for backbone
    total_grad_norm = 0.0
    for name, p in self.backbone.named_parameters():
      if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_grad_norm += param_norm.item() ** 2
        # Log individual parameter gradient norm
        self.log(f'train/grad_norm/{name}', param_norm.item(), on_step=True, on_epoch=False, sync_dist=True)

    total_grad_norm = total_grad_norm ** 0.5
    # Log total gradient norm
    self.log('train/backbone_grad_norm', total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits
  
  def _scdd_parameterization(self, logits):

    logits[:, :, self.mask_index] += self.neg_infinity
    # Return LOG probabilities (like D3PM) for numerical stability
    # This avoids information loss from softmax underflow
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits
    

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'scdd':
      return self._scdd_parameterization(logits=logits)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb


  def _get_scdd_schedule(self, t, max_ratio, gamma_shape, t_peak=0.5):
    """SCDD forward process schedule."""
    a = gamma_shape * t_peak
    b = gamma_shape * (1 - t_peak)

    base_peak = (t_peak ** a) * ((1 - t_peak) ** b)
    B = (max_ratio / (1 - max_ratio)) / base_peak
    ct = B * torch.pow(t, a) * torch.pow(1 - t, b)

    if self.config.debug:
      ct = 0 * ct

    clean_mass = (1 - t) / (1 + ct)
    uniform_mass = ct / (1 + ct)
    absorbed_mass = 1 - clean_mass - uniform_mass

    gamma_t = clean_mass + uniform_mass
    rho_t = clean_mass / gamma_t.clamp(min=1e-30)

    return ScddSchedule(
      clean_mass=clean_mass,
      uniform_mass=uniform_mass,
      absorbed_mass=absorbed_mass,
      gamma_t=gamma_t,
      rho_t=rho_t)


  def _get_scdd_transition(self, t, max_ratio, gamma_shape, dt=None, t_peak=0.5):
    if dt is None:
      dt = 1 / self.T
    s = t - dt

    schedule_t = self._get_scdd_schedule(t, max_ratio, gamma_shape, t_peak)
    schedule_s = self._get_scdd_schedule(s, max_ratio, gamma_shape, t_peak)

    clean_transition = schedule_t.clean_mass / schedule_s.clean_mass
    uniform_transition = schedule_t.gamma_t / schedule_s.gamma_t - clean_transition

    return clean_transition, uniform_transition


  def _scdd_correction_loss(self,
    rho_t, rho_s,
    clean_fraction_transition,
    uniform_fraction_transition,
    x0,
    x_t,
    model_output,
    log_num,
    sum_log_num_excl_mask,
    log_num_x0
    ):

    B, L, V = model_output.shape
    dtype = model_output.dtype
    K = V - 1  # sum domain excludes the mask token

    base_s = (1 - rho_s) / K
    base_t = (1 - rho_t) / K
    uniform_transition_per_token = uniform_fraction_transition / K

    log_num_xt = log_num.gather(dim=-1, index=x_t.unsqueeze(-1)).squeeze(-1) # (B,L)

    log_p_xt = model_output.gather(dim=-1, index=x_t.unsqueeze(-1)).squeeze(-1)  # (B,L)

    base_t_is_zero = (base_t < 1e-30)
    if base_t_is_zero.all():
      log_denom_theta = torch.log(rho_t[:, None].clamp(min=1e-30)) + log_p_xt
    else:
      log_base = torch.log(base_t)
      if log_base.ndim == 1:
        log_base = log_base[:, None]
      log_clean_p = torch.log(rho_t[:, None].clamp(min=1e-30)) + log_p_xt
      log_denom_theta = torch.logaddexp(
        log_base.expand_as(log_clean_p),
        log_clean_p
      )

    sum_log_ratio = sum_log_num_excl_mask - K * log_denom_theta  # (B,L)
    log_ratio_x0  = log_num_x0 - log_denom_theta                 # (B,L)
    log_ratio_xt  = log_num_xt - log_denom_theta                 # (B,L)

    eq_xt_x0 = (x_t == x0).to(dtype)  # (B,L)
    denom_x = (base_t[:, None] + rho_t[:, None] * eq_xt_x0) # (B,L)

    inv_denom_x = 1.0 / denom_x.clamp(min=1e-30)

    # Expand sum_z (A + B*1[z=x0]) (C + D*1[z=x_t]) log_ratio(z).
    # A = (1-rho_s) / K, B = rho_s, C = ((rho_s-rho_t)/rho_s) / K, D = rho_t/rho_s
    A = base_s[:, None]
    Bterm = rho_s[:, None]
    C = uniform_transition_per_token[:, None]
    Dterm = clean_fraction_transition[:, None]

    total = (A * C) * sum_log_ratio \
          + (Bterm * C) * log_ratio_x0 \
          + (A * Dterm) * log_ratio_xt \
          + (Bterm * Dterm * eq_xt_x0) * log_ratio_x0

    out = -inv_denom_x * total  # (B,L)

    return out


  def _scdd_loss(self, model_output, xt, x0, t):
    """
    model_output: (batch_size, seq_len, vocab_size)
    xt: (batch_size, seq_len)
    x0: (batch_size, seq_len)
    t: (batch_size, 1)
    """
    dt = 1 / self.T

    s = t - dt

    max_ratio = self.config.forward.ratio
    gamma_shape = self.config.forward.gamma
    t_peak = self.config.forward.t_peak

    schedule_t = self._get_scdd_schedule(t, max_ratio, gamma_shape, t_peak)
    schedule_s = self._get_scdd_schedule(s, max_ratio, gamma_shape, t_peak)

    gamma_t = schedule_t.gamma_t
    rho_t = schedule_t.rho_t
    gamma_s = schedule_s.gamma_t
    rho_s = schedule_s.rho_t

    rho_s_safe = rho_s.clamp(min=1e-30)
    clean_fraction_transition = rho_t / rho_s_safe
    uniform_fraction_transition = (rho_s - rho_t) / rho_s_safe

    mask_coeff = (gamma_s - gamma_t) / (1 - gamma_t)
    base = (1 - rho_s) / (self.vocab_size - 1)  # (B)

    log_p_theta = model_output  # Already in log-space! (B, L, V)
    log_rho_s = torch.log(rho_s.clamp(min=1e-30))  # (B,)

    base_is_zero = (base < 1e-30).all()  # Scalar boolean

    if base_is_zero:
      log_term = log_rho_s[:, None, None] + log_p_theta  # (B, L, V)
    else:
      log_base = torch.log(base)  # (B,)
      log_clean_p = log_rho_s[:, None, None] + log_p_theta  # (B, L, V)
      log_term = torch.logaddexp(
        log_base[:, None, None].expand_as(log_clean_p),
        log_clean_p
      )

    log_term_for_mask = log_term.clone()
    log_term_for_mask[:, :, self.mask_index] = 0

    sum_log = log_term_for_mask.sum(dim=-1)  # (B,L)
    log_at_x0 = log_term_for_mask.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B,L)

    mask_loss = -mask_coeff[:, None] * (base[:, None] * sum_log + rho_s[:, None] * log_at_x0)

    correction_loss = self._scdd_correction_loss(
      rho_t, rho_s, clean_fraction_transition, uniform_fraction_transition,
      x0, xt, model_output, log_term,
      sum_log, log_at_x0)


    flag = (xt == self.mask_index)

    loss = torch.where(flag, mask_loss, correction_loss)

    return loss * self.T


  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    


    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, mask_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      mask_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < mask_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def q_xt_sc(self, x, mask_chance, uniform_chance):
    """Compute the noisy sample xt given masking and uniform distribution.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      mask_chance: float torch.Tensor with shape (batch_size, 1), absorbed_mass_t
      uniform_chance: float torch.Tensor with shape (batch_size, 1),
          (vocab_size - 2) * uniform_mass_t / (vocab_size - 1)
    """

    # Sample mask and uniform indices
    mask_chance = mask_chance[:, None]
    uniform_chance = uniform_chance[:, None]

    random =  torch.rand(* x.shape, device=x.device)
    mask_indices = random < mask_chance
    uniform_indices = (random < mask_chance + uniform_chance) & (random >= mask_chance)

    xt = torch.where(mask_indices, self.mask_index, x)

    # Replace uniform_indices positions with vocab-uniform tokens
    uniform_probs = torch.ones((x.shape[0], x.shape[1], self.vocab_size), device=x.device) / (self.vocab_size-2)
    uniform_probs[:, :, self.mask_index] = 0
    uniform_probs.scatter_(-1, x.unsqueeze(-1), 0.0)
    uniform_tokens = _sample_categorical(uniform_probs)

    xt = torch.where(uniform_indices, uniform_tokens, xt)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    mask_chance_t = t[:, None, None]
    mask_chance_s = (t - dt)[:, None, None]
    assert mask_chance_t.ndim == 3, mask_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
      if self.config.sampling.nucleus_p < 1:
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
    
    assert mask_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (mask_chance_t - mask_chance_s)
    q_xs[:, :, self.mask_index] = mask_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    mask_chance_t = 1 - torch.exp(-sigma_t)
    mask_chance_s = 1 - torch.exp(-sigma_s)
    mask_chance_t = mask_chance_t[:, None, None]
    mask_chance_s = mask_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert mask_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (mask_chance_t
                             - mask_chance_s)
    q_xs[:, :, self.mask_index] = mask_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _llada_update(self, x, t, dt): # currently not working, will lead to ill gen_ppl
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    mask_chance_t = 1 - torch.exp(-sigma_t)
    mask_chance_s = 1 - torch.exp(-sigma_s)
    mask_chance_t = mask_chance_t[:, None, None]
    mask_chance_s = mask_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)

    # Sample candidate tokens x0 from the model's output distribution
    candidate_x0 = _sample_categorical(log_p_x0.exp())

    # Calculate Confidence
    p_x0 = log_p_x0.exp()
    confidence = torch.gather(p_x0, -1, candidate_x0.unsqueeze(-1)).squeeze(-1) # (B, L)

    # Determine Number of Tokens to Unmask
    seq_len = x.shape[1]
    target_mask_count = (seq_len * mask_chance_s.squeeze(-1)).long() # (B, 1)
    
    # Current actual mask count
    is_masked = (x == self.mask_index)
    current_mask_count = is_masked.sum(dim=-1, keepdim=True) # (B, 1)

    # Number of tokens to unmask this step
    num_to_unmask = current_mask_count - target_mask_count
    num_to_unmask = num_to_unmask.clamp(min=0)

    # Select Top-K
    # Mask out confidence of already unmasked tokens in-place
    confidence[~is_masked] = float('-inf')

    for b in range(x.shape[0]):
        k = num_to_unmask[b].item()
        if k <= 0:
            continue
        
        _, top_indices = torch.topk(confidence[b], k=k)
        
        # Update state: Replace MASK with candidate_x0 at these indices
        x[b, top_indices] = candidate_x0[b, top_indices]

    return x

  def _scdd_update(self, x, t, dt):
    """
    Posterior sampling for self-correction diffusion language model

    Args:
      x: torch.Tensor(batch_size, seq_len), current token sequence
      t: torch.Tensor(batch_size), current time step
      dt: time step difference
    """
    x_input = x
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    # Compute the sampling distribution:
    unet_conditioning = sigma_t
    log_x_theta = self.forward(x, unet_conditioning)
    # Convert from log probabilities to probabilities for sampling
    x_theta = log_x_theta.exp()

    if self.config.sampling.nucleus_p < 1.0:  # static config — no graph break
      sorted_probs, sorted_indices = torch.sort(x_theta, descending=True, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
      top_p_mask[..., 0] = True
      nucleus_probs = sorted_probs * top_p_mask
      nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
      x_theta = torch.zeros_like(x_theta).scatter_(-1, sorted_indices, nucleus_probs)

    # Compute effective times.
    eff_t = 1 - torch.exp(-sigma_t)
    eff_s = 1 - torch.exp(-sigma_s)

    # Ensure effective times are non-negative to avoid NaNs
    eff_t = eff_t.clamp(min=0.0)
    eff_s = eff_s.clamp(min=0.0)

    gamma_shape = self.config.forward.gamma
    t_peak = self.config.forward.t_peak
    max_ratio = self.config.forward.ratio

    schedule_t = self._get_scdd_schedule(eff_t, max_ratio, gamma_shape, t_peak)
    schedule_s = self._get_scdd_schedule(eff_s, max_ratio, gamma_shape, t_peak)

    gamma_t = schedule_t.gamma_t
    clean_mass_t = schedule_t.clean_mass
    uniform_mass_t = schedule_t.uniform_mass
    absorbed_mass_t = schedule_t.absorbed_mass

    gamma_s = schedule_s.gamma_t
    clean_mass_s = schedule_s.clean_mass
    uniform_mass_s = schedule_s.uniform_mass
    absorbed_mass_s = schedule_s.absorbed_mass

    clean_transition = clean_mass_t / clean_mass_s
    uniform_transition = gamma_t / gamma_s - clean_transition
    absorbing_transition = 1 - clean_transition - uniform_transition

    # Compute the posterior distribution
    # Case 1: sampling when current token is not mask
    denominator = uniform_mass_t[:, None, None] / (self.vocab_size - 1) + clean_mass_t[:, None, None] * torch.gather(x_theta, -1, x.unsqueeze(-1))

    numerator = clean_mass_s[:, None, None] * uniform_transition[:, None, None] / (self.vocab_size - 1) * x_theta \
      + uniform_mass_s[:, None, None] * uniform_transition[:, None, None] / (self.vocab_size - 1)**2

    add_x = clean_mass_s[:, None, None] * clean_transition[:, None, None] * torch.gather(x_theta, -1, x.unsqueeze(-1)) \
      + clean_transition[:, None, None] * uniform_mass_s[:, None, None] / (self.vocab_size - 1)

    numerator = numerator.scatter_add(-1, x.unsqueeze(-1), add_x)
    numerator[:, :, self.mask_index] = 0.0

    correct_probs = numerator / denominator

    # Case 2: sampling when current token is mask
    mask_nonabsorbed_coeff = absorbing_transition / absorbed_mass_t
    mask_probs = (mask_nonabsorbed_coeff[:, None, None] * uniform_mass_s[:, None, None] / (self.vocab_size-1)) \
      + (mask_nonabsorbed_coeff[:, None, None] * clean_mass_s[:, None, None]) * x_theta
    mask_probs[:, :, self.mask_index] = (absorbed_mass_s / absorbed_mass_t)[:, None]

    copy_flag = (x != self.mask_index)
    probs = torch.where(copy_flag.unsqueeze(-1), correct_probs, mask_probs)

    x = _sample_categorical(probs)

    if getattr(self, '_disable_corrections', False):
      was_unmasked = (x_input != self.mask_index)
      x = torch.where(was_unmasked, x_input, x)

    return x

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  def compile_sampler(self):
    """Compile the SCDD denoising step for faster sampling."""
    if self.sampler == 'scdd':
      self._compiled_scdd_step = torch.compile(
        ScddDenoisingStep(self))
    return self

  def _should_compile_sampler(self):
    sampling_config = getattr(self.config, 'sampling', None)
    if sampling_config is None:
      return True
    if hasattr(sampling_config, 'get'):
      return sampling_config.get('compile_sampler', True)
    return getattr(sampling_config, 'compile_sampler', True)

  def _maybe_compile_sampler(self):
    if (self.sampler == 'scdd'
        and self._should_compile_sampler()
        and not hasattr(self, '_compiled_scdd_step')):
      self.compile_sampler()

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model.
    
    Returns:
      If config.eval.track_corrections is True, returns (samples, correction_stats)
      Otherwise, returns samples only.
    """
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      result = self._ar_sampler(batch_size_per_gpu)
      if getattr(self.config.eval, 'track_corrections', False):
        return result, {}
      return result
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    # Initialize correction statistics accumulator
    track_corrections = getattr(self.config.eval, 'track_corrections', False)
    track_per_step = getattr(self.config.eval, 'track_per_step', False)
    correction_stats = {
      'total_tokens': 0,
      'total_changed': 0,
      'mask_to_nonmask': 0,
      'nonmask_to_nonmask': 0,
      'corrections': 0,
      'still_mask': 0,
      'unchanged': 0,
    } if track_corrections else None
    per_step_stats = [] if (track_corrections and track_per_step) else None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      
      x_prev = x
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'scdd':
        if hasattr(self, '_compiled_scdd_step'):
          x = self._compiled_scdd_step(x, t, dt)
        else:
          x = self._scdd_update(x, t, dt)
      elif self.sampler == 'llada':
        x = self._llada_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)
      
      # Track corrections if enabled
      if track_corrections:
        step_stats = self._check_token_changes(x_prev, x, i)
        
        # Add time information to step stats if tracking per-step
        if track_per_step:
          step_stats_with_time = step_stats.copy()
          step_stats_with_time['step_index'] = i
          step_stats_with_time['step_time'] = i  # Step index as time
          step_stats_with_time['relative_time'] = i / max(num_steps, 1)  # Relative time (0-1) based on steps
          per_step_stats.append(step_stats_with_time)
        
        # Accumulate totals
        for key in correction_stats:
          if key in step_stats:
            correction_stats[key] += step_stats[key]
      elif self.config.eval.print_changes:
        self._check_token_changes(x_prev, x, i)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
    
    if track_corrections:
      if track_per_step:
        return x, {'total': correction_stats, 'per_step': per_step_stats}
      else:
        return x, correction_stats
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model.
    
    Returns:
      If config.eval.track_corrections is True, returns (samples, correction_stats)
      Otherwise, returns samples only.
    """
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    self._maybe_compile_sampler()
    result = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return result

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
      t = t.clamp(0., 1.-1e-4)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      mask_chance = torch.exp(f_0 + t * (f_T - f_0))
      mask_chance = mask_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      mask_chance = 1 - torch.exp(-sigma[:, None])

    if self.config.forward.name == 'mix':

      if self.config.debug:
        xt = self.q_xt(x0, mask_chance)
      else:
        schedule_t = self._get_scdd_schedule(
          t, self.config.forward.ratio,
          self.config.forward.gamma, self.config.forward.t_peak)
        mask_chance = schedule_t.absorbed_mass
        uniform_chance = (
          (self.vocab_size - 2) * schedule_t.uniform_mass
          / (self.vocab_size-1))

        xt = self.q_xt_sc(x0, mask_chance, uniform_chance)
        
    else: 
      xt = self.q_xt(x0, mask_chance)
    
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:

      if  self.parameterization == 'scdd':
        diffusion_loss = self._scdd_loss(model_output=model_output, xt=xt, x0=x0, t=t)
      else:
        diffusion_loss = self._d3pm_loss(model_output=model_output, xt=xt, x0=x0, t=t)

      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization in {'subs', 'scdd'}:
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
