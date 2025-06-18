import logging
import os
import torch

from pathlib import Path
from omegaconf import OmegaConf

import prism_embedder.distributed as distributed
from prism_embedder.utils import fix_random_seeds, setup_logging
from prism_embedder.configs import default_config

OUTPUT_PATH = Path("/output")
logger = logging.getLogger("prism_embedder")


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_file(config_file):
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.resolve(cfg)
    return cfg


def setup(config_file):
    """
    Basic configuration setup without any distributed or GPU-specific initialization.
    This function:
      - Loads the config from file and command-line options.
      - Sets up logging.
      - Fixes random seeds.
      - Creates the output directory.
    """
    cfg = get_cfg_from_file(config_file)
    output_dir = OUTPUT_PATH
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = str(output_dir)

    fix_random_seeds(0)
    setup_logging(output=output_dir, level=logging.INFO)
    return cfg


def setup_distributed():
    """
    Distributed/GPU setup. This function handles:
      - Enabling distributed mode.
      - Distributed logging, seeding adjustments based on rank
    """
    distributed.enable(overwrite=True)

    torch.distributed.barrier()

    # update random seed using rank
    rank = distributed.get_global_rank()
    fix_random_seeds(rank)
