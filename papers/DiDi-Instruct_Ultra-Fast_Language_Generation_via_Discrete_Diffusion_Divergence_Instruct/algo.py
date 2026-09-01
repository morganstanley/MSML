import collections
import copy
import math
import os
import pickle
import sys

import fsspec
import lightning as L
import numpy as np
import scipy.stats as stats
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.callbacks.progress import TQDMProgressBar

import trainer_base
import utils


class DisabledProgressBar(ProgressBar):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): pass
    def on_train_epoch_start(self, trainer, pl_module): pass
    def on_validation_start(self, trainer, pl_module): pass
    def on_test_start(self, trainer, pl_module): pass
    def on_predict_start(self, trainer, pl_module): pass
    def disable(self): return True
    def enable(self): return False
    def is_enabled(self): return True

class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()
    if self.config.mode == "sample_eval":
       self._loaded_checkpoint_path = self.config.eval.checkpoint_path
    self.print_every = getattr(config.algo, 'print_every', 10)
    self.enable_progress_bar = getattr(config.algo, 'enable_progress_bar', True)
    self.current_print = -1

  def on_train_start(self):
    save_dir = self.trainer.default_root_dir
    if self.trainer.checkpoint_callback and hasattr(self.trainer.checkpoint_callback, 'dirpath'):
      save_dir = self.trainer.checkpoint_callback.dirpath
    self.print(f"----------------------------------------------------------------------------------------------------\n"
               f"  --> Checkpoint save to: \n"
               f"  --> \"{save_dir}\"\n"
               f"----------------------------------------------------------------------------------------------------\n")
    filepath = os.path.join(save_dir, "student_before_train.ckpt")
    os.makedirs(save_dir, exist_ok=True)
    self.trainer.save_checkpoint(filepath)
    
  def _validate_configuration(self):
    pass

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
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

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def training_step(self, batch, batch_idx):
      loss = super().training_step(batch, batch_idx)
      if self.print_every and self.global_step % self.print_every == 0 and self.global_step > 0 and self.current_print != self.global_step:
          self.current_print = self.global_step
          self.print(f"Global Step: {self.global_step}, Train Loss: {loss.item():.4f}")
      return loss

class DiDiInstruct(MDLM):
  """
  Implements DiDi-Instruct to distill a few-step generator.
  """
  class DiscriminatorHead(nn.Module):
    """
    A classification head for the DIT model to act as a discriminator.
    It takes the sequence of hidden states and outputs a logit PER TOKEN.
    """
    def __init__(self, hidden_size):
      super().__init__()
      self.main = nn.Sequential(
        torch.nn.utils.spectral_norm(nn.Linear(in_features=hidden_size, out_features=hidden_size)),
        nn.SiLU(),
        torch.nn.utils.spectral_norm(nn.Linear(in_features=hidden_size, out_features=1))
      )
    def forward(self, x, c=None):
      return self.main(x)
    
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)

    # --- Model Setup ---
    self.teacher = copy.deepcopy(self.backbone)
    self.teacher.eval()
    for param in self.teacher.parameters():
        param.requires_grad = False
    
    # --- Discriminator ---
    self.discriminator = copy.deepcopy(self.backbone)
    for param in self.discriminator.parameters():
        param.requires_grad = False
    
    num_blocks_to_unfreeze = getattr(config.algo, 'discriminator_unfreeze_blocks', 4)
    if num_blocks_to_unfreeze > 0 and hasattr(self.discriminator, 'blocks'):
        for block in self.discriminator.blocks[-num_blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True
    if hasattr(self.discriminator, 'final_layer'):
        for param in self.discriminator.final_layer.parameters():
            param.requires_grad = True
    if hasattr(self.discriminator, 'norm'):
        for param in self.discriminator.norm.parameters():
            param.requires_grad = True
    self.discriminator.output_layer = self.DiscriminatorHead(config.model.hidden_size)
    
    # EMA student
    self.student_ema = copy.deepcopy(self.backbone)
    self.student_ema.eval()
    for param in self.student_ema.parameters():
        param.requires_grad = False
      
    # Hyperparameters
    self.discriminator_lr = getattr(config.algo, 'discriminator_lr', 1e-5)
    self.kl_beta = getattr(config.algo, 'kl_beta', 1.00)
    self.entropy_beta = getattr(config.algo, 'entropy_beta', 0.02)
    self.sampling_eps = getattr(config.algo, 'sampling_eps', 0.02)
    self.label_smoothing = getattr(config.algo, 'label_smoothing', 0.1)
    self.discriminator_warmup_steps = getattr(config.algo, 'discriminator_warmup_steps', 0)
    self.student_update_every = getattr(config.algo, 'student_update_every', 1) 
    self.remask_prob = getattr(config.algo, 'remask_prob', 0.0)
    self.reward_clip_val = getattr(config.algo, 'reward_clip_val', 8.0)
    self.log_odds_eps = getattr(config.algo, 'log_odds_eps', 1e-6)
    self.num_samples_per_prompt = getattr(config.algo, 'num_samples_per_prompt', 4)
    self.advantage_clip_val = getattr(config.algo, 'advantage_clip_val', 3.0)
    self.grad_clip_value = getattr(config.algo, 'gradient_clip_val', 1.0)
    self.omega_min = getattr(config.algo, 'omega_min', 0.1)
    self.omega_max = getattr(config.algo, 'omega_max', 10.0)
    
    # Hyperparameters
    two_step_config = getattr(config.algo, 'two_step', {})
    self.tau_mode = getattr(config.algo, 'tau_mode', 'beta22')
    self.tau_min_eps = two_step_config.get('tau_min_eps', 0.02)
    self.t0_eps = two_step_config.get('t0_eps', 1e-3)
    self.logprob_mode = two_step_config.get('logprob_mode', 'prev_is_mask')
    self.kl_hi_coef = two_step_config.get('kl_hi_coef', 0.01)
    self.kl_lo_coef = two_step_config.get('kl_lo_coef', 0.01)
    self.importance_weight = getattr(config.algo, 'importance_weight', False)
    self.num_candidates = getattr(config.algo, 'num_candidates', 1)
    self.guidance_scale_start = getattr(config.algo, 'guidance_scale_start', 0.2)
    self.guidance_scale_end = getattr(config.algo, 'guidance_scale_end', 1.0)
    self.rerank_steps_ratio = getattr(config.algo, 'rerank_steps_ratio', 0.5)

    # Logging & Config
    self.print_every = getattr(config.algo, 'print_every', 10)
    self.enable_progress_bar = getattr(config.algo, 'enable_progress_bar', False)
    self.current_print = -1
    self.optim_config_student = config.optim
    self.optim_config_discriminator = config.algo.discriminator_optim
    self.lr_scheduler_config = config.algo.lr_scheduler
    self.save_hyperparameters()
    self.save_model_every = config.algo.save_after_n_steps
    self.last_growth_step = -1
    self.out_dir = getattr(config.algo, 'output_dir', None)
    self.global_steps = 0
    self.ema_beta = getattr(config.algo, 'ema_beta', 0.999)
    self.register_buffer('reward_baseline', torch.tensor(0.0))

  def update_ema(self):
    with torch.no_grad():
      for param, ema_param in zip(self.backbone.parameters(), self.student_ema.parameters()):
        ema_param.data.lerp_(param.data, 1.0 - self.ema_beta)

  def on_train_start(self):
    if not self.enable_progress_bar:
      for i, callback in enumerate(self.trainer.callbacks):
        if isinstance(callback, TQDMProgressBar):
          self.trainer.callbacks[i] = DisabledProgressBar()
          break
    save_dir = self.trainer.default_root_dir
    if self.trainer.checkpoint_callback and hasattr(self.trainer.checkpoint_callback, 'dirpath'):
      save_dir = self.trainer.checkpoint_callback.dirpath
    self.print(f"\n----------------------------------------------------------------------------------------------------\n"
               f"  --> Checkpoint save to: \n"
               f"  --> \"{save_dir}\"\n"
               f"----------------------------------------------------------------------------------------------------\n")
    os.makedirs(save_dir, exist_ok=True)

  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.teacher.to(*args, **kwargs)
    self.discriminator.to(*args, **kwargs)
    self.student_ema.to(*args, **kwargs)
    return self

  def _eval_mode(self):
    self.eval()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher')
    )
    super(MDLM, self).on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher')
    )
    super(MDLM, self).on_load_checkpoint(checkpoint)

  def load_state_dict(self, state_dict, strict=False):
    return super().load_state_dict(state_dict, strict=strict)
    
  def _get_lr(self):
    step = self.global_steps
    opt_student, opt_discriminator = self.optimizers()
    warmup_steps = self.lr_scheduler_config.warmup_steps
    total_steps = self.trainer.max_steps

    def get_lr(peak_lr):
      if step < warmup_steps:
          return peak_lr * (step + 1) / warmup_steps
      progress = (step - warmup_steps) / (total_steps - warmup_steps)
      return max(peak_lr * (1.0 - progress), 1e-8)

    for param_group in opt_student.param_groups:
        param_group['lr'] = get_lr(self.optim_config_student.lr)
    for param_group in opt_discriminator.param_groups:
        param_group['lr'] = get_lr(self.discriminator_lr)

    return opt_student, opt_discriminator
  
  def _get_tau(self, batch_size):
    """
    Samples tau, the intermediate timestep for the two-step trajectory.
    Also computes the corresponding alpha and sigma values.
    """
    r = torch.rand(batch_size, device=self.device)
    pi_tau = torch.ones_like(r)

    if self.tau_mode.startswith("beta"):
      a = float(self.tau_mode[4]) if len(self.tau_mode) > 4 else 2.0
      b = float(self.tau_mode[5]) if len(self.tau_mode) > 5 else a
      beta_dist = torch.distributions.Beta(torch.tensor(a, device=self.device), torch.tensor(b, device=self.device))
      tau = beta_dist.sample((batch_size,))
      pi_tau = beta_dist.log_prob(tau).exp()
    else: tau = r 
    
    tau = tau.clamp(min=self.tau_min_eps, max=1.0 - self.sampling_eps)
    
    _, alpha_tau = self.noise(tau)
    sigma_tau = self._sigma_from_alphat(alpha_tau.unsqueeze(-1)).squeeze(-1)
    
    return tau, alpha_tau, sigma_tau, pi_tau
  
  def _corrupt_with_mask_ratio(self, x0, alpha_t):
      masking_prob = 1.0 - alpha_t.unsqueeze(-1)
      noise = torch.rand_like(x0, dtype=torch.float32)
      mask = noise < masking_prob
      zt = x0.clone()
      zt[mask] = self.mask_index
      return zt, mask

  def _get_masked_input(self, batch_size):
    shape = (batch_size, self.num_tokens)
    return torch.full(shape, self.mask_index, device=self.device, dtype=torch.long)
  
  def _get_corrupted_inputs(self, batch_size, tau_gen=None, alpha_gen=None, sigma_gen=None):
      if tau_gen is None:
          raise ValueError("tau_gen, alpha_gen, and sigma_gen must be provided for coupled trajectory mode.")
      return tau_gen, alpha_gen, sigma_gen
  
  @torch.no_grad()
  def _get_rewards(self, zt_fake, sigma_corr, mask_fake, original_batch_size, G):
    with torch.amp.autocast('cuda', enabled=False):
        discriminator_logits_on_fake = self.discriminator(zt_fake, sigma=sigma_corr)
        probs_d = torch.sigmoid(discriminator_logits_on_fake)
        log_odds = torch.log(probs_d + self.log_odds_eps) - torch.log(1 - probs_d + self.log_odds_eps)
        clipped_log_odds = torch.clamp(log_odds, -self.reward_clip_val, self.reward_clip_val)
        reward_per_token = clipped_log_odds.squeeze(-1)
        masked_reward = reward_per_token.masked_fill(~mask_fake, 0)
        reward = masked_reward.sum(dim=-1) / (mask_fake.sum(dim=-1) + 1e-8)
        reward_mean_preshift = reward.mean()
        reward_std_preshift = reward.std()
        is_clipped = (clipped_log_odds.squeeze(-1) != log_odds.squeeze(-1)).float()
        reward_fraction_clipped = is_clipped[mask_fake].mean()
        advantage = torch.zeros_like(reward)
        
        if self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            gathered_rewards_list = [torch.zeros_like(reward) for _ in range(world_size)]
            dist.all_gather(gathered_rewards_list, reward)
            global_rewards_flat = torch.cat(gathered_rewards_list, dim=0)
            global_rewards_groups = global_rewards_flat.view(-1, G)
            group_mean_global = global_rewards_groups.mean(dim=1, keepdim=True)
            group_std_global = global_rewards_groups.std(dim=1, keepdim=True)
            rank = dist.get_rank()
            local_slice_start = rank * original_batch_size
            local_slice_end = (rank + 1) * original_batch_size
            group_mean_local = group_mean_global[local_slice_start:local_slice_end]
            group_std_local = group_std_global[local_slice_start:local_slice_end]
            local_reward_groups = reward.view(original_batch_size, G)
            advantage = (local_reward_groups - group_mean_local) / (group_std_local + 1e-8)
        else:
            reward_groups = reward.view(original_batch_size, G)
            group_mean = reward_groups.mean(dim=1, keepdim=True)
            group_std = reward_groups.std(dim=1, keepdim=True)
            advantage = (reward_groups - group_mean) / (group_std + 1e-8)
        final_reward = torch.clamp(advantage, -self.advantage_clip_val, self.advantage_clip_val).view(-1)
        reward_mean = reward_mean_preshift
        reward_std = reward_std_preshift
    return final_reward, reward_mean, reward_std, reward_fraction_clipped, advantage

  def _get_backward(self, x_prev, t_hi, t_lo):
    batch_size = x_prev.shape[0]
    if torch.is_tensor(t_hi):
        t_tensor_hi = t_hi.squeeze()
    else:
        t_tensor_hi = torch.full((batch_size,), t_hi, device=self.device)
    if torch.is_tensor(t_lo):
        t_lo = t_lo.squeeze()
    dt = t_tensor_hi - t_lo
    p_x0, x_next = self._get_ancestral_update(x=x_prev, t=t_tensor_hi, dt=dt)
    logits = torch.log(p_x0 + 1e-10)
    log_probs = F.log_softmax(logits, dim=-1)
    log_q_sampled = torch.gather(log_probs, 2, x_next.unsqueeze(-1)).squeeze(-1)
    prev_was_mask = (x_prev == self.mask_index)
    if self.logprob_mode == "changed_only":
        newly_unmasked = prev_was_mask & (x_next != self.mask_index)
        logp_step = (log_q_sampled * newly_unmasked).sum(dim=-1)
    else:
        logp_step = (log_q_sampled * prev_was_mask).sum(dim=-1)
    return x_next, logp_step, logits, prev_was_mask
  
  def _get_regularization(self, batch_size, x_T, x_tau, sigma_gen, logits_step1, mask_step1, 
                          logits_step2, mask_step2):
    with torch.no_grad():
        sigma_hi = self._sigma_from_alphat(self.noise(torch.full((batch_size,), 1.0, device=self.device))[1].unsqueeze(-1)).squeeze(-1)
        teacher_logits_step1 = self.teacher(x_T, sigma=sigma_hi)
    with torch.no_grad():
        teacher_logits_step2 = self.teacher(x_tau, sigma=sigma_gen)
    kl_loss_step1 = F.kl_div(F.log_softmax(logits_step1, dim=-1), F.softmax(teacher_logits_step1, dim=-1), reduction='none', log_target=False).sum(dim=-1)
    kl_loss_step2 = F.kl_div(F.log_softmax(logits_step2, dim=-1), F.softmax(teacher_logits_step2, dim=-1), reduction='none', log_target=False).sum(dim=-1)
    loss_kl_hi = (kl_loss_step1 * mask_step1).sum() / (mask_step1.sum() + 1e-8)
    loss_kl_lo = (kl_loss_step2 * mask_step2).sum() / (mask_step2.sum() + 1e-8)
    loss_kl = self.kl_hi_coef * loss_kl_hi + self.kl_lo_coef * loss_kl_lo
    entropy_step1 = torch.distributions.Categorical(logits=logits_step1).entropy()
    entropy_step1 = (entropy_step1 * mask_step1).sum() / (mask_step1.sum() + 1e-8)
    entropy_step2 = torch.distributions.Categorical(logits=logits_step2).entropy()
    entropy_step2 = (entropy_step2 * mask_step2).sum() / (mask_step2.sum() + 1e-8)
    entropy = (entropy_step1 + entropy_step2) / 2
    return loss_kl, entropy

  @torch.no_grad()
  def _get_ancestral_update(self, x, t, dt, p_x0=None, noise_removal_step=False):
    _, alpha_t = self.noise(t)
    if noise_removal_step:
        alpha_s = torch.ones_like(alpha_t)
    else:
        _, alpha_s = self.noise(t - dt)
    
    if p_x0 is None:
        if alpha_t.ndim == 1:
            alpha_t_2d = alpha_t.unsqueeze(-1)
        else:
            alpha_t_2d = alpha_t
        with torch.enable_grad():
            sigma = self._sigma_from_alphat(alpha_t_2d)
            p_x0 = self.forward(x, sigma).exp()
            p_x0 = p_x0.detach()
    q_xs = p_x0 * (alpha_s - alpha_t)[:, None, None]
    q_xs[:, :, self.mask_index] = (1 - alpha_s)[:, None]
    _x = trainer_base.sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    with torch.enable_grad():
        sigma_recompute = self._sigma_from_alphat(alpha_t.unsqueeze(-1) if alpha_t.ndim == 1 else alpha_t)
        p_x0_for_grad = self.forward(x, sigma_recompute).exp()
    return p_x0_for_grad, copy_flag * x + (1 - copy_flag) * _x

  def _get_gradient_tilting_step(self, x, t, dt, guidance_scale, discriminator):
    sigma = self._sigma_from_alphat(self.noise(t)[1])
    log_p_x0 = self.forward(xt=x, sigma=sigma)
    if guidance_scale == 0:
        _, alpha_t = self.noise(t)
        _, alpha_s = self.noise(t - dt)
        p_x0 = log_p_x0.exp()
        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        sampled_tokens = trainer_base.sample_categorical(q_xs)
        return torch.where(x != self.mask_index, x, sampled_tokens)
    masked_pos = (x == self.mask_index)
    with torch.enable_grad():
        log_p_x0_for_grad = log_p_x0.clone().requires_grad_()
        unmasked_one_hot = F.one_hot(x.clamp(min=0), self.vocab_size).float() * 1e9
        input_logits_for_disc = torch.where(masked_pos.unsqueeze(-1), log_p_x0_for_grad, unmasked_one_hot)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if sigma.ndim == 2:
                sigma = sigma.squeeze(-1)
            discriminator_scores = discriminator(input_logits_for_disc, sigma)
        reward = (discriminator_scores.squeeze(-1) * masked_pos).sum()
        reward.backward()
        grad = log_p_x0_for_grad.grad
    grad_masked = torch.where(masked_pos.unsqueeze(-1), grad, torch.zeros_like(grad))
    grad_clipped = grad_masked.clamp_(-1.0, 1.0)
    tilted_log_p_x0 = log_p_x0 + guidance_scale * grad_clipped.detach()
    p_x0_guided = tilted_log_p_x0.softmax(dim=-1)
    _, alpha_t = self.noise(t)
    _, alpha_s = self.noise(t - dt)
    q_xs = p_x0_guided * (alpha_s - alpha_t)[:, :, None]
    q_xs[:, :, self.mask_index] = 1 - alpha_s
    sampled_tokens = trainer_base.sample_categorical(q_xs)
    return torch.where(x != self.mask_index, x, sampled_tokens)
  
  def _get_guided_ancestral_step(self, x, t, dt, guidance_scale, discriminator):
    sigma = self._sigma_from_alphat(self.noise(t)[1])
    log_p_x0 = self.forward(xt=x, sigma=sigma)
    if guidance_scale == 0 or self.num_candidates == 1:
        _, alpha_t = self.noise(t)
        _, alpha_s = self.noise(t - dt)
        p_x0 = log_p_x0.exp()
        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        sampled_tokens = trainer_base.sample_categorical(q_xs)
        return torch.where(x != self.mask_index, x, sampled_tokens)
    batch_size, seq_len = x.shape
    p_x0 = log_p_x0.exp()
    _, alpha_t = self.noise(t)
    _, alpha_s = self.noise(t - dt)
    q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
    q_xs[:, :, self.mask_index] = 1 - alpha_s
    q_xs_expanded = q_xs.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
    sampled_tokens = trainer_base.sample_categorical(q_xs_expanded)
    copy_flag = (x != self.mask_index).unsqueeze(1)
    x_candidates = torch.where(copy_flag, x.unsqueeze(1), sampled_tokens)
    sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
    if sigma_s.ndim == 2:
        sigma_s = sigma_s.squeeze(-1)
    x_candidates_flat = x_candidates.view(batch_size * self.num_candidates, seq_len)
    sigma_s_expanded = sigma_s.repeat_interleave(self.num_candidates, dim=0)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        discriminator_scores_flat = discriminator(x_candidates_flat, sigma_s_expanded)
    reward_flat = discriminator_scores_flat.sum(dim=[1, 2])
    rewards = reward_flat.view(batch_size, self.num_candidates)
    rerank_probs = torch.softmax(rewards * guidance_scale, dim=-1)
    best_candidate_indices = torch.multinomial(rerank_probs, num_samples=1).squeeze(-1)
    best_indices_expanded = best_candidate_indices.view(batch_size, 1, 1).expand(-1, -1, seq_len)
    x_next = torch.gather(x_candidates, 1, best_indices_expanded).squeeze(1)
    return x_next
  
  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
    if self.config.mode == "sample_eval":
        print("\n========================================================")
        print(f"Generating samples from model:")
        if hasattr(self, '_loaded_checkpoint_path'):
            print(f"  --> {self._loaded_checkpoint_path}")
        print("========================================================")
    if num_steps is None:
      num_steps = self.config.sampling.steps
    self.backbone.eval()
    self.discriminator.eval()
    x = self.prior_sample(num_samples, self.num_tokens)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    rerank_start_step = int(num_steps * (1 - self.rerank_steps_ratio))
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
      if self.config.mode == "sample_eval":
          print(f"[Eval] Backward step {i+1}/{num_steps}, t = {t.mean().item():.4f}")
      if 'ancestral' in self.sampler:
        _, x = super(MDLM, self)._ancestral_update(x=x, t=t, dt=dt, p_x0=None)
      elif self.sampler == 'analytic':
        x = super(MDLM, self)._analytic_update(x=x, t=t, dt=dt)
      elif self.sampler == 'guided':
        progress = i / (num_steps - 1)
        guid_scale = self.guidance_scale_start + progress * (self.guidance_scale_end - self.guidance_scale_start)
        if i < rerank_start_step:
            x = self._get_gradient_tilting_step(x, t, dt, guid_scale, self.discriminator)
        else:
            x = self._get_guided_ancestral_step(x, t, dt, guid_scale, self.discriminator)
      else:
        raise ValueError(f"Unknown sampler: {self.sampler}")
    t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
    if self.config.sampling.noise_removal == 'ancestral':
        x = self._denoiser_update(x=x, t=t0)
    elif self.config.sampling.noise_removal == 'greedy':
      sigma = self._sigma_from_alphat(self.noise(t0)[1])
      model_output = self.forward(xt=x, sigma=sigma)
      x = self._process_model_output(model_output, xt=x, sigma=sigma).argmax(dim=-1)
    self.backbone.train()
    self.discriminator.train()
    return x
  
  def training_step(self, batch, batch_idx):
      step = self.global_steps
      opt_student, opt_discriminator = self._get_lr()
      original_batch_size = batch['input_ids'].shape[0]
      G = self.num_samples_per_prompt
      repeated_input_ids = batch['input_ids'].repeat_interleave(G, dim=0)
      batch_size = repeated_input_ids.shape[0]

      # Student Generation
      tau_gen, _, sigma_gen, pi_tau = self._get_tau(batch_size)
      x_T = self._get_masked_input(batch_size)
      x_tau, logp_step1, logits_step1, mask_step1 = self._get_backward(x_T, 1.0, tau_gen)
      if self.remask_prob > 0:
          unmasked_in_step1 = mask_step1 & (x_tau != self.mask_index)
          remask_noise = torch.rand_like(x_tau, dtype=torch.float32)
          should_remask = remask_noise < self.remask_prob
          remask_mask = unmasked_in_step1 & should_remask
          x_tau[remask_mask] = self.mask_index
      x0_fake, logp_step2, logits_step2, mask_step2 = self._get_backward(x_tau, tau_gen, self.t0_eps)
      log_q_theta_x0 = logp_step1 + logp_step2
      x0_real = repeated_input_ids
      tau_corr, alpha_corr, sigma_corr = self._get_corrupted_inputs(batch_size, tau_gen, None, sigma_gen)
      zt_fake, mask_fake = self._corrupt_with_mask_ratio(x0_fake, alpha_corr)
      zt_real, mask_real = self._corrupt_with_mask_ratio(x0_real, alpha_corr)
      opt_discriminator.zero_grad()

      real_logits = self.discriminator(zt_real, sigma=sigma_corr)
      fake_logits = self.discriminator(zt_fake.detach(), sigma=sigma_corr)
      real_labels = torch.full_like(real_logits, 1.0 - self.label_smoothing)
      fake_labels = torch.full_like(fake_logits, 0.0 + self.label_smoothing)
      
      bce_r = F.binary_cross_entropy_with_logits(real_logits, real_labels, reduction="none").squeeze(-1)
      bce_f = F.binary_cross_entropy_with_logits(fake_logits, fake_labels, reduction="none").squeeze(-1)
      loss_d_real = (bce_r * mask_real).sum() / (mask_real.sum() + 1e-8)
      loss_d_fake = (bce_f * mask_fake).sum() / (mask_fake.sum() + 1e-8)
      loss_d = (loss_d_real + loss_d_fake) / 2

      self.manual_backward(loss_d)
      disc_grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in self.discriminator.parameters() if p.grad is not None))
      for param in self.discriminator.parameters():
          if param.grad is not None:
              torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
      if self.grad_clip_value > 0:
          torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_value)
      opt_discriminator.step()
      with torch.no_grad():
          real_preds = (real_logits > 0).float()
          fake_preds = (fake_logits < 0).float()
          disc_acc = (real_preds.sum() + fake_preds.sum()) / (real_logits.numel() + fake_logits.numel())

      # Student Update
      loss_pg, loss_kl, entropy, student_grad_norm = 0, 0, 0, torch.tensor(0)
      update_student = False
      if step >= self.discriminator_warmup_steps:
          if step % self.student_update_every == 0:
              update_student = True
      if update_student:
          opt_student.zero_grad()
          final_reward, reward_mean, reward_std, reward_fraction_clipped, advantage = self._get_rewards(
              zt_fake=zt_fake,
              sigma_corr=sigma_corr,
              mask_fake=mask_fake,
              original_batch_size=original_batch_size,
              G=G
          )
          with torch.no_grad():
              eps = 1e-4
              _, alpha_tau_minus_eps = self.noise(torch.clamp(tau_gen - eps, min=0.0))
              alpha_prime_tau = (alpha_gen - alpha_tau_minus_eps) / eps
              omega_t = -alpha_prime_tau / (1 - alpha_gen + 1e-8)
              omega_t = torch.clamp(omega_t, min=self.omega_min, max=self.omega_max)
          if self.importance_weight:
              omega_t_ = omega_t / (pi_tau + 1e-8)
          else:
              omega_t_ = omega_t
          loss_pg = - (omega_t_.detach() * final_reward.detach() * log_q_theta_x0).mean()
          loss_kl, entropy = self._get_regularization(
              batch_size=batch_size,
              x_T=x_T,
              x_tau=x_tau,
              sigma_gen=sigma_gen,
              logits_step1=logits_step1,
              mask_step1=mask_step1,
              logits_step2=logits_step2,
              mask_step2=mask_step2
          )
          total_student_loss = loss_pg + self.kl_beta * loss_kl - self.entropy_beta * entropy
          self.manual_backward(total_student_loss)
          student_grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in self.backbone.parameters() if p.grad is not None))
          for param in self.backbone.parameters():
              if param.grad is not None:
                  torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
          if self.grad_clip_value > 0:
              torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.grad_clip_value*0.5)
          opt_student.step()
          self.update_ema()
          loss_pg, loss_kl, entropy = loss_pg.item(), loss_kl.item(), entropy.item()

      # Logging
      self.log('trainer/lr_student', opt_student.param_groups[0]['lr'], on_step=True, on_epoch=False, sync_dist=True)
      self.log('trainer/lr_discriminator', opt_discriminator.param_groups[0]['lr'], on_step=True, on_epoch=False, sync_dist=True)
      self.log('trainer/loss_discriminator', loss_d.item(), on_step=True, on_epoch=False, sync_dist=True)
      self.log('monitor/discriminator_acc', disc_acc.item(), on_step=True, on_epoch=False, sync_dist=True)
      self.log('monitor/grad_norm_discriminator', disc_grad_norm.item(), on_step=True, on_epoch=False, sync_dist=True)
      if update_student:
          self.log('trainer/loss_student_pg', loss_pg, on_step=True, on_epoch=False, sync_dist=True)
          self.log('trainer/loss_student_entropy_bonus', entropy, on_step=True, on_epoch=False, sync_dist=True)
          total_student_loss = loss_pg + loss_kl - self.entropy_beta * entropy
          self.log('trainer/loss_student_total', total_student_loss, on_step=True, on_epoch=False, sync_dist=True)
          self.log('monitor/grad_norm_student', student_grad_norm.item(), on_step=True, on_epoch=False, sync_dist=True)

      if self.print_every and step > 0 and step % self.print_every == 0 and self.current_print != step:
          self.current_print = step
          max_logits = -1.0
          with torch.no_grad():
              if 'logits_step2' in locals() and logits_step2 is not None:
                  max_logits = logits_step2.max(dim=-1).values.mean().item()
              elif 'logits_step1' in locals() and logits_step1 is not None:
                  max_logits = logits_step1.max(dim=-1).values.mean().item()
          self.log('monitor/max_logits', max_logits, on_step=True, on_epoch=False, sync_dist=True)
          self.print(
              f"  --> Steps: {step}, "
              f"D Loss: {loss_d.item():.3e}, "
              f"G Loss: {loss_pg:.3e}, "
              f"Entropy: {entropy:.3f}, "
              f"D Acc: {disc_acc.item():.3f}, "
              f"Grad (S|D): {student_grad_norm.item():.2e}|{disc_grad_norm.item():.2e}, "
              f"Max Logits: {max_logits:.2e}"
          )

      if (step % self.save_model_every == 0) and (step != self.last_growth_step) and step >= 0:
        self.last_growth_step = step
        save_dir = self.trainer.default_root_dir
        if self.trainer.checkpoint_callback and hasattr(self.trainer.checkpoint_callback, 'dirpath'):
          save_dir = self.trainer.checkpoint_callback.dirpath
        filename = f"student_iters_{step}.ckpt"
        filepath = os.path.join(save_dir, filename)
        short_path = os.path.relpath(filepath, self.out_dir) if self.out_dir else filepath
        self.print(f"================================================================================================\n"
                  f"  --> Saving checkpoint to: \n"
                  f"  --> \"/{short_path}\"\n"
                  f"================================================================================================")
        self.trainer.save_checkpoint(filepath)

      self.global_steps += 1
  
  def configure_optimizers(self):
    optimizer_student = torch.optim.AdamW(
      self.backbone.parameters(),
      lr=self.optim_config_student.lr,
      weight_decay=self.optim_config_student.weight_decay,
      betas=(self.optim_config_student.beta1, self.optim_config_student.beta2)
    )
    trainable_discriminator_params = filter(
        lambda p: p.requires_grad, self.discriminator.parameters()
    )
    optimizer_discriminator = torch.optim.AdamW(
      trainable_discriminator_params,
      lr=self.discriminator_lr,
      betas=(self.optim_config_discriminator.beta1, self.optim_config_discriminator.beta2),
      weight_decay=self.optim_config_discriminator.weight_decay
    )
    return [optimizer_student, optimizer_discriminator]
  
