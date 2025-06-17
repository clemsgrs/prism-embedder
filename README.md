# prism-embedder

[![PyPI version](https://img.shields.io/pypi/v/prism-embedder?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/prism-embedder/)
[![Docker Version](https://img.shields.io/docker/v/waticlems/prism_embedder?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/prism_embedder)


## 🛠️ Installation

System requirements: Linux-based OS (e.g., Ubuntu 22.04) with Python 3.10+ and Docker installed.

We recommend running the script inside a container using the latest `prism_embedder` image from Docker Hub:

```shell
docker pull waticlems/prism_embedder:latest
docker run --rm -it \
    -v /path/to/your/slide.tif:/input/images/whole-slide-image/slide.tif \
    -v /path/to/your/mask.tif:/input/images/tissue-mask/mask.tif \
    waticlems/prism_embedder:latest
```

Update the command with the path pointing to your slide & tissue mask.

Alternatively, you can install `prism-embedder` via pip:

```shell
pip install prism-embedder
```

## TODO

- [] save feature in `.json` compatible with GC interface
- [] update interface slug based on new interface request (whole-slide image and whole-slide tiling visualization)
- [] plot tSNE for a few slides to debug implementation