"""Implementation based on nanoGPT (https://github.com/karpathy/nanoGPT)
"""
from typing import Any

import chex
import flax.linen as nn
import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class GPT2Config:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    dtype: Any = jnp.float32
    use_ln: bool = True
    use_linear_attention: bool = False


# Linear weights and Embedding weights are initialized with mean 0, stddev 0.02 normal random variables.
# Linear biases are initialized to 0 which is the default.
# Residual projection weights get scaled by a factor of 1/sqrt(2 * n_layer)
# Layer norm weights are initialized to 1 (default), biases are initialized to 0 (default), and epsilon is set to 1e-5
init_fn = nn.initializers.normal(0.02)
get_scaled_init_fn = lambda n_layer: nn.initializers.normal(0.02 / jnp.sqrt(2 * n_layer))


class GPT2SelfAttention(nn.Module):
    config: GPT2Config

    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=init_fn)
        self.c_proj = nn.Dense(
            self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=get_scaled_init_fn(self.config.n_layer)
        )
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_droput = nn.Dropout(self.config.dropout)

    def __call__(self, x: Array, attention_mask: Array, training: bool = False) -> Array:
        B, T, C = x.shape  # batch_size, block_size, n_embd
        q, k, v = jnp.split(self.c_attn(x), 3, axis=2)
        q = q.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = k.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        
        att = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(k.shape[-1])  # (B, nh, T, T)
        # attention_mask shape: (B, T, T) -> expand to (B, 1, T, T) for broadcasting across heads
        mask = attention_mask[:, None, :, :]  # Add head dimension
        att = jnp.where(mask, att, jnp.finfo(self.config.dtype).min)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not training)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_droput(self.c_proj(y), deterministic=not training)
        return y


class GPT2LinearSelfAttentionOld(nn.Module):
    """Original linear attention implementation - preserved for reference"""
    config: GPT2Config

    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=init_fn)
        self.c_proj = nn.Dense(
            self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=get_scaled_init_fn(self.config.n_layer)
        )
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_droput = nn.Dropout(self.config.dropout)

    def __call__(self, x: Array, attention_mask: Array, training: bool = False) -> Array:
        B, T, C = x.shape  # batch_size, block_size, n_embd
        q, k, v = jnp.split(self.c_attn(x), 3, axis=2)
        q = q.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = k.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        
        # ELU feature maps: φ(x) = ELU(x) + 1
        phi_q = nn.elu(q) + 1.0  # (B, nh, T, hs)
        phi_k = nn.elu(k) + 1.0  # (B, nh, T, hs)
        
        # Apply attention mask by zeroing out masked positions
        # For linear attention, we only need to mask the key/value positions
        # attention_mask shape: (B, T, T), we need the diagonal for valid positions
        # Reduce (B, T, T) to (B, T) — keep a key if *any* query attends to it
        # Note: linear attention cannot use the full attention mask like in standard attention
        # or otherwise we would lose teh speed advantage of linear attention.
        valid_keys = jnp.any(attention_mask > 0, axis=1).astype(self.config.dtype)  # (B, T)
        mask = valid_keys[:, None, :, None]  # (B, 1, T, 1)

        phi_k = phi_k * mask.astype(self.config.dtype)
        v = v * mask.astype(self.config.dtype)
        
        # Linear attention: φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)
        # phi_k: (B, nh, T, hs), v: (B, nh, T, hs)
        kv = jnp.einsum('bhtd,bhtf->bhdf', phi_k, v)  # (B, nh, hs, hs)
        k_sum = jnp.sum(phi_k, axis=2, keepdims=True)  # (B, nh, 1, hs)
        
        # phi_q: (B, nh, T, hs), kv: (B, nh, hs, hs)
        numerator = jnp.einsum('bhtd,bhdf->bhtf', phi_q, kv)  # (B, nh, T, hs)
        # phi_q: (B, nh, T, hs), k_sum: (B, nh, 1, hs)
        denominator = jnp.einsum('bhtd,bhrd->bhtr', phi_q, k_sum)  # (B, nh, T, 1)
        
        # Avoid division by zero
        denominator = jnp.maximum(denominator, 1e-8)
        y = numerator / denominator  # (B, nh, T, hs)
        
        y = self.attn_dropout(y, deterministic=not training)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_droput(self.c_proj(y), deterministic=not training)
        return y


class GPT2LinearSelfAttention(nn.Module):
    """Corrected linear attention implementation with chex assertions"""
    config: GPT2Config

    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=init_fn)
        self.c_proj = nn.Dense(
            self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=get_scaled_init_fn(self.config.n_layer)
        )
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_droput = nn.Dropout(self.config.dropout)

    def __call__(self, x: Array, attention_mask: Array, training: bool = False) -> Array:
        B, T, C = x.shape  # batch_size, block_size, n_embd
        chex.assert_shape(x, (B, T, C))
        chex.assert_shape(attention_mask, (B, T, T))
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        chex.assert_shape(qkv, (B, T, 3 * C))
        
        q, k, v = jnp.split(qkv, 3, axis=2)
        chex.assert_shape(q, (B, T, C))
        chex.assert_shape(k, (B, T, C))
        chex.assert_shape(v, (B, T, C))
        
        # Reshape for multi-head attention
        head_dim = C // self.config.n_head
        q = q.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = k.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        chex.assert_shape(q, (B, self.config.n_head, T, head_dim))
        chex.assert_shape(k, (B, self.config.n_head, T, head_dim))
        chex.assert_shape(v, (B, self.config.n_head, T, head_dim))
        
        # ELU feature maps: φ(x) = ELU(x) + 1
        phi_q = nn.elu(q) + 1.0  # (B, nh, T, hs)
        phi_k = nn.elu(k) + 1.0  # (B, nh, T, hs)
        chex.assert_shape(phi_q, (B, self.config.n_head, T, head_dim))
        chex.assert_shape(phi_k, (B, self.config.n_head, T, head_dim))
        
        # Apply attention mask (padding) by zeroing out invalid positions
        # attention_mask shape: (B, T, T)
        # Extract valid positions from attention mask - position i is valid if attention_mask[i, i] > 0
        valid_positions = jnp.diagonal(attention_mask, axis1=1, axis2=2)  # (B, T)
        chex.assert_shape(valid_positions, (B, T))
        
        position_mask = valid_positions[:, None, :, None]  # (B, 1, T, 1)
        chex.assert_shape(position_mask, (B, 1, T, 1))
        
        # Mask the features and values at invalid positions
        phi_k = phi_k * position_mask
        v = v * position_mask
        chex.assert_shape(phi_k, (B, self.config.n_head, T, head_dim))
        chex.assert_shape(v, (B, self.config.n_head, T, head_dim))
        
        # Causal linear attention using cumulative sums
        # For each position t, we want: sum_{s=1}^t φ(k_s) ⊗ v_s / sum_{s=1}^t φ(k_s)
        
        # Cumulative sum for numerator: φ(K)^T @ V up to position t
        # phi_k: (B, nh, T, hs), v: (B, nh, T, hs)
        kv_outer = jnp.einsum('bhti,bhtj->bhtij', phi_k, v)  # (B, nh, T, hs, hs)
        chex.assert_shape(kv_outer, (B, self.config.n_head, T, head_dim, head_dim))
        
        kv_cumsum = jnp.cumsum(kv_outer, axis=2)  # (B, nh, T, hs, hs)
        chex.assert_shape(kv_cumsum, (B, self.config.n_head, T, head_dim, head_dim))
        
        # Cumulative sum for denominator: φ(K)^T @ 1 up to position t  
        k_cumsum = jnp.cumsum(phi_k, axis=2)  # (B, nh, T, hs)
        chex.assert_shape(k_cumsum, (B, self.config.n_head, T, head_dim))
        
        # Compute output: φ(Q) @ cumsum(φ(K)^T @ V) / (φ(Q) @ cumsum(φ(K)^T @ 1))
        # phi_q: (B, nh, T, hs), kv_cumsum: (B, nh, T, hs, hs)
        numerator = jnp.einsum('bhti,bhtij->bhtj', phi_q, kv_cumsum)  # (B, nh, T, hs)
        chex.assert_shape(numerator, (B, self.config.n_head, T, head_dim))
        
        # phi_q: (B, nh, T, hs), k_cumsum: (B, nh, T, hs)  
        denominator = jnp.einsum('bhti,bhti->bht', phi_q, k_cumsum)  # (B, nh, T)
        chex.assert_shape(denominator, (B, self.config.n_head, T))
        
        # Avoid division by zero
        denominator = jnp.maximum(denominator, 1e-8)
        chex.assert_shape(denominator, (B, self.config.n_head, T))
        
        y = numerator / denominator[..., None]  # (B, nh, T, hs)
        chex.assert_shape(y, (B, self.config.n_head, T, head_dim))
        
        y = self.attn_dropout(y, deterministic=not training)
        chex.assert_shape(y, (B, self.config.n_head, T, head_dim))
        
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        chex.assert_shape(y, (B, T, C))
        
        y = self.resid_droput(self.c_proj(y), deterministic=not training)
        chex.assert_shape(y, (B, T, C))
        
        return y


class GPT2MLP(nn.Module):
    config: GPT2Config

    def setup(self):
        self.c_fc = nn.Dense(4 * self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=init_fn)
        self.c_proj = nn.Dense(
            self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=get_scaled_init_fn(self.config.n_layer)
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def __call__(self, x: Array, training: bool = False) -> Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not training)
        return x


class GPT2Block(nn.Module):
    config: GPT2Config

    def setup(self):
        if self.config.use_ln:
            self.ln_1 = nn.LayerNorm(1e-5, self.config.dtype, use_bias=self.config.bias)
        if self.config.use_linear_attention:
            self.attn = GPT2LinearSelfAttention(self.config)
        else:
            self.attn = GPT2SelfAttention(self.config)
        if self.config.use_ln:
            self.ln_2 = nn.LayerNorm(1e-5, self.config.dtype, use_bias=self.config.bias)
        self.mlp = GPT2MLP(self.config)

    def __call__(self, x: Array, attention_mask: Array, training: bool = False) -> Array:
        if self.config.use_ln:
            x = x + self.attn(self.ln_1(x), attention_mask, training=training)
            x = x + self.mlp(self.ln_2(x), training=training)
        else:
            x = x + self.attn(x, attention_mask, training=training)
            x = x + self.mlp(x, training=training)
        return x


class GPT2Model(nn.Module):
    config: GPT2Config

    def setup(self):
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd, self.config.dtype, embedding_init=init_fn)
        self.drop = nn.Dropout(self.config.dropout)
        self.hs = [GPT2Block(self.config) for _ in range(self.config.n_layer)]
        if self.config.use_ln:
            self.ln_f = nn.LayerNorm(1e-5, self.config.dtype, use_bias=self.config.bias)

    def __call__(self, input_embds: Array, attention_mask: Array, training: bool = False) -> Array:
        pos = jnp.expand_dims(jnp.arange(self.config.block_size), axis=0)  # (1, T)
        pos_embds = self.wpe(pos)  # (1, T, n_embd)
        x = input_embds + pos_embds  #  (B, T, n_embd)
        x = self.drop(x, deterministic=not training)
        for h in self.hs:
            x = h(x, attention_mask, training=training)
        if self.config.use_ln:
            x = self.ln_f(x)
        return x
