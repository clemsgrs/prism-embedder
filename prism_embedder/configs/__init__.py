from pathlib import Path
from omegaconf import OmegaConf

CONFIG_DIR = Path("/opt/app/prism_embedder/configs")


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(CONFIG_DIR.resolve() / config_filename)


default_config = load_config("default")


def load_and_merge_config(config_name: str):
    default_config = OmegaConf.create(default_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)
