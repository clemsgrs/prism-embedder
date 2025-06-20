import gc
import os
import numpy as np
import tqdm
import torch
import argparse
import traceback
import torchvision
import pandas as pd
import multiprocessing as mp
import wholeslidedata as wsd

from pathlib import Path
from contextlib import nullcontext

import prism_embedder.distributed as distributed

from prism_embedder.utils import fix_random_seeds, write_json_file
from prism_embedder.utils.config import get_cfg_from_file
from prism_embedder.models import PRISM

torchvision.disable_beta_transforms_warning()

MODEL_DIR = Path("/opt/ml/model")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("PRISM Embedder", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    return parser


def scale_coordinates(wsi_fp, coordinates, spacing, backend):
    """
    Scale coordinates based on the target spacing.
    """
    wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
    min_spacing = wsi.spacings[0]
    scale = min_spacing / spacing
    scaled_coordinates = (coordinates * scale).astype(int)
    return scaled_coordinates


def main(args):
    # setup configuration
    cfg = get_cfg_from_file(args.config_file)

    coordinates_dir = Path(cfg.output_dir, "coordinates")
    fix_random_seeds(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers_embedding)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    process_list = Path(cfg.output_dir, "process_list.csv")
    assert (
        process_list.is_file()
    ), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)
    skip_feature_aggregation = process_df["aggregation_status"].str.contains("success").all()

    if skip_feature_aggregation and distributed.is_main_process():
        print("Feature aggregation already completed.")
        return

    model = PRISM(MODEL_DIR)
    model.eval()
    model = model.to(model.device)

    # select slides where tile-level feature extraction was successfull
    tiled_df = process_df[process_df.tiling_status == "success"]
    tiled_and_features_df = tiled_df[tiled_df.embedding_status == "success"]
    mask = tiled_and_features_df["aggregation_status"] != "success"
    process_stack = tiled_and_features_df[mask]
    total = len(process_stack)
    wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]

    features_dir = Path(cfg.output_dir, "features")

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.speed.fp16
        else nullcontext()
    )
    feature_aggregation_updates = {}

    for wsi_fp in tqdm.tqdm(
        wsi_paths_to_process,
        desc="Pooling tile features",
        unit="slide",
        total=total,
        leave=True,
    ):
        try:

            name = wsi_fp.stem.replace(" ", "_")
            coordinates_file = coordinates_dir / f"{name}.npy"
            coordinates_arr = np.load(coordinates_file, allow_pickle=True)
            coordinates = (np.array([coordinates_arr["x"], coordinates_arr["y"]]).T).astype(int)

            feature_path = features_dir / f"{name}.pt"

            coordinates = torch.tensor(
                coordinates,
                dtype=torch.int,
                device=model.device,
            )

            with torch.inference_mode():
                with autocast_context:
                    features = torch.load(feature_path).to(model.device)
                    wsi_feature = model.forward(features).squeeze(0).cpu().tolist()

            output_dict = [{"title": name, "features": wsi_feature}]
            output_filename = Path(cfg.output_dir) / "image-neural-representation.json"
            write_json_file(
                location=output_filename,
                content=output_dict,
            )
            print(f"Feature vector saved to {output_filename}")

            del wsi_feature
            torch.cuda.empty_cache()
            gc.collect()

            feature_aggregation_updates[str(wsi_fp)] = {"status": "success"}

        except Exception as e:
            feature_aggregation_updates[str(wsi_fp)] = {
                "status": "failed",
                "error": str(e),
                "traceback": str(traceback.format_exc()),
            }

        # update process_df
        status_info = feature_aggregation_updates[str(wsi_fp)]
        process_df.loc[
            process_df["wsi_path"] == str(wsi_fp), "aggregation_status"
        ] = status_info["status"]
        if "error" in status_info:
            process_df.loc[
                process_df["wsi_path"] == str(wsi_fp), "error"
            ] = status_info["error"]
            process_df.loc[
                process_df["wsi_path"] == str(wsi_fp), "traceback"
            ] = status_info["traceback"]
        process_df.to_csv(process_list, index=False)

    # summary logging
    slides_with_tile_features = len(tiled_and_features_df)
    total_slides = len(process_df)
    failed_feature_aggregation = process_df[
        ~(process_df["aggregation_status"] == "success")
    ]
    print("=+=" * 10)
    print(f"Total number of slides with tile-level features: {slides_with_tile_features}/{total_slides}")
    print(f"Failed slide-level feature aggregation: {len(failed_feature_aggregation)}")
    print(
        f"Completed slide-level feature aggregation: {total_slides - len(failed_feature_aggregation)}"
    )
    print("=+=" * 10)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
