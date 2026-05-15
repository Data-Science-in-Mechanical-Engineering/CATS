import collections
import copy
from functools import partial
import gc
import itertools
import logging
import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import time
import numpy as np
import omegaconf
import optax
from tqdm import tqdm

import hydra

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
import equinox as eqx

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray 

import torch

from model.timeseries_decoder import TimeseriesPatchedDecoder
from model.vit import VIT
import utils.jax_utils as ju

import mpx

from dataset_loaders.dataset_utils import init_tf_dataloader_timeseries, init_torch_dataloader
from execution_guard import execution_guard

from trainer import Trainer

def train(cfg: omegaconf.DictConfig, logger):
    my_trainer = Trainer(cfg, logger)
    my_trainer.train(pruning_step=cfg.pruning_step)

@hydra.main(config_path="parameters", config_name="main", version_base="1.1")
@partial(execution_guard, force_overwrite=True)
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger("execution_guard")
    train(cfg, logger)


if __name__ == "__main__":
    main()
