import json
import timm
import torch
import logging
import torch.nn as nn

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from pathlib import Path

import prism_embedder.distributed as distributed

from prism_embedder.models.prism.configuring_prism import (
    PerceiverConfig,
    PrismConfig,
)
from prism_embedder.models.prism.modeling_prism import Prism
from transformers.models.biogpt.configuration_biogpt import BioGptConfig

from prism_embedder.utils import update_state_dict

logger = logging.getLogger("prism_embedder")


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.encoder = self.build_encoder()
        self.set_device()
        self.load_weights()

    def set_device(self):
        if distributed.is_enabled():
            self.device = torch.device(f"cuda:{distributed.get_local_rank()}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def build_encoder(self):
        raise NotImplementedError

    def load_weights(self):
        raise NotImplementedError

    def get_transforms(self):
        data_config = resolve_data_config(
            self.encoder.pretrained_cfg, model=self.encoder
        )
        transform = create_transform(**data_config)
        return transform

    def forward(self, x):
        raise NotImplementedError


class Virchow(FeatureExtractor):
    """
    Tile-level feature extractor.
    """

    def __init__(self, model_dir, input_size: int, mode: str = "cls"):
        self.model_dir = model_dir
        assert input_size == 224, "Virchow model best works with 224px tiles"
        self.mode = mode
        self.features_dim = 1280
        if mode == "full":
            self.features_dim = 2560
        super(Virchow, self).__init__()

    def build_encoder(self):
        # load model configuration
        with open(Path(self.model_dir, "virchow-config.json"), "r") as f:
            self.config = json.load(f)

        encoder = timm.create_model(
            **self.config, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )
        return encoder

    def load_weights(self):
        checkpoint_path = Path(self.model_dir, "virchow.pth")
        print(f"Loading tile encoder weights from {checkpoint_path}...")
        weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=self.encoder.state_dict(), state_dict=weights
        )
        print(msg)
        self.encoder.load_state_dict(updated_sd, strict=True)

    def get_transforms(self):
        data_config = resolve_data_config(
            self.config["pretrained_cfg"], model=self.encoder
        )
        return create_transform(**data_config)

    def forward(self, x):
        output = self.encoder(x)
        class_token = output[:, 0]  # size: 1 x 1280
        patch_tokens = output[
            :, 1:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        if self.mode == "cls":
            return class_token
        elif self.mode == "full":
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: 1 x 2560
            return embedding


class SlideFeatureExtractor(nn.Module):
    def __init__(self, model_dir: Path):
        super(SlideFeatureExtractor, self).__init__()
        self.model_dir = model_dir
        self.encoder = self.build_encoder()
        self.set_device()
        self.load_weights()

    def build_encoder(self):
        raise NotImplementedError

    def load_weights(self):
        raise NotImplementedError

    def set_device(self):
        if distributed.is_enabled():
            self.device = torch.device(f"cuda:{distributed.get_local_rank()}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def forward(self, tile_features):
        raise NotImplementedError


class PRISM(SlideFeatureExtractor):

    def build_encoder(self):
        # import sys
        # sys.path.insert(0, self.model_dir)
        cfg = PrismConfig(
            biogpt_config=BioGptConfig(),
            perceiver_config=PerceiverConfig(),
            model_dir=self.model_dir,
        )
        encoder = Prism(cfg)
        return encoder

    def load_weights(self):
        checkpoint_path = Path(self.model_dir, "prism.pth")
        weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=self.encoder.state_dict(), state_dict=weights
        )
        print(msg)
        self.encoder.load_state_dict(updated_sd, strict=True)

    def forward(self, tile_features):
        tile_features = tile_features.unsqueeze(0)
        reprs = self.encoder.slide_representations(tile_features)
        output = reprs["image_embedding"]  # [1, 1280]
        return output
