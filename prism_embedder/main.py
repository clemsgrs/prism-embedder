import os
import sys
import socket
import signal
import argparse
import subprocess
import pandas as pd

from pathlib import Path

from prism_embedder.utils import setup, _show_torch_cuda_info, print_directory_contents

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("PRISM Embedder", add_help=add_help)
    parser.add_argument(
        "--config-file", default="prism_embedder/configs/prism.yaml", metavar="FILE", help="path to config file"
    )
    return parser


def generate_csv(wsi_dir, mask_dir, csv_path):

    wsi_files = sorted(list(wsi_dir.glob("*.tif")))
    mask_files = sorted(list(mask_dir.glob("*.tif")))
    df = pd.DataFrame({
        "wsi_path": [str(f) for f in wsi_files],
        "mask_path": [str(f) for f in mask_files],
    })
    df.to_csv(csv_path, index=False)


def run_tiling(config_file):
    print("Running tiling.py...")
    cmd = [
        sys.executable,
        "prism_embedder/tiling.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Slide tiling failed. Exiting.")
        sys.exit(result.returncode)


def run_feature_extraction(config_file):
    print("Running embed.py...")
    # find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        free_port = s.getsockname()[1]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--master_port={free_port}",
        "--nproc_per_node=gpu",
        "prism_embedder/embed.py",
        "--config-file",
        config_file,
    ]
    # launch in its own process group.
    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        text=True,
    )
    try:
        proc.communicate()
    except KeyboardInterrupt:
        print("Received CTRL+C, terminating embed.py process group...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        sys.exit(1)
    if proc.returncode != 0:
        print("Feature extraction failed. Exiting.")
        sys.exit(proc.returncode)


def run_feature_aggregation(config_file):
    print("Running aggregate.py...")
    # find a free port
    cmd = [
        sys.executable,
        "prism_embedder/aggregate.py",
        "--config-file",
        config_file,
    ]
    # launch in its own process group.
    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        text=True,
    )
    try:
        proc.communicate()
    except KeyboardInterrupt:
        print("Received CTRL+C, terminating embed.py process group...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        sys.exit(1)
    if proc.returncode != 0:
        print("Feature aggregation failed. Exiting.")
        sys.exit(proc.returncode)


def main(args):

    config_file = args.config_file
    cfg = setup(config_file)

    # generate csv from input folder
    # whole slide will be located under /input/images/whole-slide-image/<wsi-uuid>.tif
    # tissue mask will be located under /input/images/tissue-mask/<mask-uuid>.tif

    wsi_dir = INPUT_PATH / "images/whole-slide-image"
    mask_dir = INPUT_PATH / "images/tissue-mask"
    csv_path = cfg.csv

    generate_csv(wsi_dir, mask_dir, csv_path)

    run_tiling(config_file)
    print("Tiling completed.")
    print("=+=" * 10)

    run_feature_extraction(config_file)
    print("Feature extraction completed.")
    print("=+=" * 10)

    run_feature_aggregation(config_file)
    print("Feature aggregation completed.")
    print("=+=" * 10)

    print("All tasks finished successfully.")
    print("=+=" * 10)


def run():

    import warnings
    import torchvision

    torchvision.disable_beta_transforms_warning()

    warnings.filterwarnings("ignore", message=".*Could not set the permissions.*")
    warnings.filterwarnings("ignore", message=".*antialias.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*TypedStorage.*", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

    # show GPU information
    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)
    print("=+=" * 10)

    args = get_args_parser(add_help=True).parse_args()
    main(args)

    print_directory_contents(OUTPUT_PATH)
    print("=+=" * 10)


if __name__ == "__main__":

    raise SystemExit(run())
