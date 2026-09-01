import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile
from typing import Optional

import datasets
import fsspec
import numpy as np
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)
SEEDS = 42

def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def get_dataset(dataset_name,
                tokenizer,
                wrap,
                mode,
                cache_dir,
                insert_eos=True,
                block_size=1024,
                num_proc=len(os.sched_getaffinity(0)),
                streaming=False,
                ratio=None,
                revision : Optional[str]=None):
  eos_tag = ''
  if not insert_eos:
    eos_tag = '_eosFalse'
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped{eos_tag}.dat'
  _path = os.path.join(cache_dir, filename)

  # Synchronization Logic
  import torch.distributed as dist
  import time
  import datetime
  
  # Ensure distributed group is initialized for synchronization
  is_distributed = False
  if dist.is_available() and dist.is_initialized():
      is_distributed = True
  elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
      # Initialize distributed definition if in torchrun but not initialized
      try:
          backend = 'nccl' if torch.cuda.is_available() else 'gloo'
          if not dist.is_initialized():
             # Set long timeout (3 hours) for data generation
             timeout = datetime.timedelta(seconds=10800)
             dist.init_process_group(backend=backend, timeout=timeout)
          is_distributed = True
          LOGGER.info(f"Initialized distributed group ({backend}) for data loading sync.")
      except Exception as e:
          LOGGER.warning(f"Failed to init process group: {e}")
  
  rank = dist.get_rank() if is_distributed else 0
  
  # Rank > 0: Wait for Rank 0 -> Load -> Return
  if is_distributed and rank > 0:
    LOGGER.info(f"Rank {rank} waiting for data generation by Rank 0 at {_path}")
    dist.barrier()
    LOGGER.info(f"Rank {rank} released from barrier. Loading data from {_path}")
    
    if not utils.fsspec_exists(_path):
        LOGGER.warning(f"Rank {rank} - Data not found after barrier. Waiting briefly...")
        time.sleep(2) 
        if not utils.fsspec_exists(_path):
             raise FileNotFoundError(f"Rank {rank} expected data at {_path} after barrier, but not found.")
    
    try:
        dataset = datasets.load_from_disk(_path)
        LOGGER.info(f"Rank {rank} loaded dataset.")
        dataset = dataset.shuffle(seed=SEEDS)
        num_data = int(len(dataset) * ratio)
        dataset = dataset.select(range(num_data))
        return dataset.with_format('torch')
    except Exception as e:
        raise RuntimeError(f"Rank {rank} failed to load data after waiting: {e}")

  # Only Rank 0 (or non-distributed) checks existence first
  if utils.fsspec_exists(_path):
    try:
      dataset = datasets.load_from_disk(_path)
      # Check if it looks valid
      _ = len(dataset)
      
      # If successful and we are Rank 0, release others
      if is_distributed: 
         LOGGER.info(f"Rank 0: Data found and valid. Releasing other ranks.")
         dist.barrier()
         
      # # MODIFICATION: Shuffle the dataset
      dataset = dataset.shuffle(seed=SEEDS)
      num_data = int(len(dataset) * ratio)
      dataset = dataset.select(range(num_data))

      LOGGER.info(f'Loading data from: {_path}, name: {dataset_name}, data size: {dataset.shape}')
      return dataset.with_format('torch')
    except Exception as e:
      LOGGER.warning(f"Failed to load existing dataset at {_path}: {e}. cleaning up.")
      import shutil
      if os.path.exists(_path):
         shutil.rmtree(_path)
      # Fallthrough to generation
  
  LOGGER.info(f'Generating new data at: {_path}')
  LOGGER.info(f'{streaming=}')  
  LOGGER.info(f'data from {cache_dir}')  

  # crop_train logic removed as only openwebtext is used
  
  if dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc)
  else:
    raise ValueError(f"Unknown dataset_name: {dataset_name}. Only openwebtext is supported.")

  data = dataset

  detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      if insert_eos:
        tokens = {'input_ids':
                  [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True)
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  
  tokenized_dataset = tokenized_dataset.remove_columns('text')

  if not wrap:
    if not streaming:
      tokenized_dataset.save_to_disk(_path)
      if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        dist.barrier()
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True)
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
      dist.barrier()
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer
    

def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  if skip_train:
    train_set = None
  else:
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      insert_eos=config.data.insert_train_eos,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      ratio=config.data.get("train_ratio", 1.0),
      revision=config.data.get("train_revision", None))
  
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      insert_eos=config.data.insert_valid_eos,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      ratio=config.data.get("valid_ratio", 1.0),
      revision=config.data.get("valid_revision", None))

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
