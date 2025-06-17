import gc
import os
import h5py
import tqdm
import torch
import argparse
import traceback
import torchvision
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from contextlib import nullcontext

import prism_embedder.distributed as distributed

from prism_embedder.utils import fix_random_seeds
from prism_embedder.utils.config import get_cfg_from_file, setup_distributed
from prism_embedder.models import Virchow
from prism_embedder.data import TileDataset

torchvision.disable_beta_transforms_warning()

MODEL_DIR = Path("/opt/ml/model")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("PRISM Embedder", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    return parser


def create_dataset(wsi_fp, coordinates_dir, spacing, backend, transforms):
    return TileDataset(
        wsi_fp,
        coordinates_dir,
        spacing,
        backend=backend,
        transforms=transforms,
    )


def run_inference(dataloader, model, device, autocast_context, unit, batch_size, feature_path, feature_dim, dtype):
    with h5py.File(feature_path, "w") as f:
        features = f.create_dataset("features", shape=(0, *feature_dim), maxshape=(None, *feature_dim), dtype=dtype, chunks=(batch_size, *feature_dim))
        indices = f.create_dataset("indices", shape=(0,), maxshape=(None,), dtype='int64', chunks=(batch_size,))
        with torch.inference_mode(), autocast_context:
            for batch in tqdm.tqdm(
                dataloader,
                desc=f"Inference on GPU {distributed.get_global_rank()}",
                unit=unit,
                unit_scale=batch_size,
                leave=False,
                position=2 + distributed.get_global_rank(),
            ):
                idx, image = batch
                image = image.to(device, non_blocking=True)
                feature = model(image).cpu().numpy()
                features.resize(features.shape[0] + feature.shape[0], axis=0)
                features[-feature.shape[0]:] = feature
                indices.resize(indices.shape[0] + idx.shape[0], axis=0)
                indices[-idx.shape[0]:] = idx.cpu().numpy()

                # cleanup
                del image, feature

    # cleanup
    torch.cuda.empty_cache()
    gc.collect()


def load_and_sort_features(tmp_dir, name):
    features_list, indices_list = [], []
    for rank in range(distributed.get_global_size()):
        fp = tmp_dir / f"{name}-rank_{rank}.h5"
        with h5py.File(fp, "r") as f:
            features_list.append(torch.from_numpy(f["features"][:]))
            indices_list.append(torch.from_numpy(f["indices"][:]))
        os.remove(fp)
    features = torch.cat(features_list, dim=0)
    indices = torch.cat(indices_list, dim=0)
    sorted_indices = torch.argsort(indices)
    sorted_features = features[sorted_indices]
    return sorted_features


def main(args):
    # setup configuration
    cfg = get_cfg_from_file(args.config_file)

    setup_distributed()

    coordinates_dir = Path(cfg.output_dir, "coordinates")
    fix_random_seeds(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unit = "tile"

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers_embedding)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    process_list = Path(cfg.output_dir, "process_list.csv")
    assert (
        process_list.is_file()
    ), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)
    skip_feature_extraction = process_df["embedding_status"].str.contains("success").all()

    if skip_feature_extraction:
        if distributed.is_main_process():
            print("=+=" * 10)
            print(f"All slides have been embedded. Skipping {unit}-level feature extraction step.")
            print("=+=" * 10)
        if distributed.is_enabled():
            torch.distributed.destroy_process_group()

    else:
        model = Virchow(MODEL_DIR, input_size=cfg.model.tile_size, mode="full")
        model.eval()
        model = model.to(model.device)
        if distributed.is_main_process():
            print(f"Starting {unit}-level feature extraction...")
        torch.distributed.barrier()

        # select slides that were successfully tiled but not yet processed for feature extraction
        tiled_df = process_df[process_df.tiling_status == "success"]
        mask = tiled_df.embedding_status != "success"
        process_stack = tiled_df[mask]
        total = len(process_stack)
        wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]

        features_dir = Path(cfg.output_dir, "features")
        if distributed.is_main_process():
            features_dir.mkdir(exist_ok=True, parents=True)

        tmp_dir = Path("/tmp")
        if distributed.is_main_process():
            tmp_dir.mkdir(exist_ok=True, parents=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if cfg.speed.fp16
            else nullcontext()
        )
        feature_extraction_updates = {}

        transforms = model.get_transforms()

        for wsi_fp in tqdm.tqdm(
            wsi_paths_to_process,
            desc="Inference",
            unit="slide",
            total=total,
            leave=True,
            disable=not distributed.is_main_process(),
            position=1,
        ):
            try:
                dataset = create_dataset(wsi_fp, coordinates_dir, cfg.tiling.params.spacing, cfg.tiling.backend, transforms)
                if distributed.is_enabled_and_multiple_gpus():
                    sampler = torch.utils.data.DistributedSampler(
                        dataset,
                        shuffle=False,
                        drop_last=False,
                    )
                else:
                    sampler = None
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=cfg.model.batch_size,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                )

                name = wsi_fp.stem.replace(" ", "_")
                feature_path = features_dir / f"{name}.pt"
                tmp_feature_path = tmp_dir / f"{name}-rank_{distributed.get_global_rank()}.h5"

                # get feature dimension and dtype using a dry run
                with torch.inference_mode(), autocast_context:
                    sample_batch = next(iter(dataloader))
                    sample_image = sample_batch[1].to(model.device)
                    sample_feature = model(sample_image).cpu().numpy()
                    feature_dim = sample_feature.shape[1:]
                    dtype = sample_feature.dtype

                run_inference(
                    dataloader,
                    model,
                    model.device,
                    autocast_context,
                    unit,
                    cfg.model.batch_size,
                    tmp_feature_path,
                    feature_dim,
                    dtype,
                )

                torch.distributed.barrier()

                if distributed.is_main_process():
                    wsi_feature = load_and_sort_features(tmp_dir, name)
                    torch.save(wsi_feature, feature_path)

                    # cleanup
                    del wsi_feature
                    torch.cuda.empty_cache()
                    gc.collect()

                torch.distributed.barrier()

                feature_extraction_updates[str(wsi_fp)] = {"status": "success"}

            except Exception as e:
                feature_extraction_updates[str(wsi_fp)] = {
                    "status": "failed",
                    "error": str(e),
                    "traceback": str(traceback.format_exc()),
                }

            # update process_df
            if distributed.is_main_process():
                status_info = feature_extraction_updates[str(wsi_fp)]
                process_df.loc[
                    process_df["wsi_path"] == str(wsi_fp), "embedding_status"
                ] = status_info["status"]
                if "error" in status_info:
                    process_df.loc[
                        process_df["wsi_path"] == str(wsi_fp), "error"
                    ] = status_info["error"]
                    process_df.loc[
                        process_df["wsi_path"] == str(wsi_fp), "traceback"
                    ] = status_info["traceback"]
                process_df.to_csv(process_list, index=False)

        if distributed.is_enabled_and_multiple_gpus():
            torch.distributed.barrier()

        if distributed.is_main_process():
            # summary logging
            slides_with_tiles = len(tiled_df)
            total_slides = len(process_df)
            failed_feature_extraction = process_df[
                ~(process_df["embedding_status"] == "success")
            ]
            print("=+=" * 10)
            print(f"Total number of slides with {unit}s: {slides_with_tiles}/{total_slides}")
            print(f"Failed {unit}-level feature extraction: {len(failed_feature_extraction)}")
            print(
                f"Completed {unit}-level feature extraction: {total_slides - len(failed_feature_extraction)}"
            )
            print("=+=" * 10)

        if distributed.is_enabled():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
