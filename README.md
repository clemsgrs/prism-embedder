<h1 align="center">PRISM Embedder</h1>

<p align="center">
  <img src="prism-embedder-logo.png" alt="PRISM Embedder Logo" width="200"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/prism-embedder/">
    <img src="https://img.shields.io/pypi/v/prism-embedder?label=pypi&logo=pypi&color=3776AB" alt="PyPI version"/>
  </a>
  <a href="https://hub.docker.com/r/waticlems/prism_embedder">
    <img src="https://img.shields.io/docker/v/waticlems/prism_embedder?sort=semver&label=docker&logo=docker&color=2496ED" alt="Docker version"/>
  </a>
</p>

<p align="center">
  Extract slide-level embeddings from whole-slide images using PRISM.
</p>

---
## 👁️ Overview

**PRISM Embedder** is a containerized algorithm for computing slide-level representations from whole-slide images using [PRISM](https://huggingface.co/paige-ai/Prism).

> ⚠️ **Access Required**  
> This algorithm depends on the gated models **PRISM** and **Virchow**.  
> You need to have **requested and been granted** access to both models on Hugging Face.


## 🚀 Quickstart

**Requirements:**  
- Linux-based OS (e.g., Ubuntu 22.04)  
- Python 3.10+  
- Docker installed

### 1. Clone the repository

```bash
git clone https://github.com/clemsgrs/prism-embedder.git
cd prism-embedder
```

### 2. Download model weights

Download the required model weights:

```bash
./download_weights.sh
```

### 3. Kick-off woth Docker

> Replace `</path/to/your/slide.tif>` and `</path/to/your/mask.tif>` with your actual file paths.  
> Replace `</path/to/your/output/folder>` with the path to a folder where you have **write access**.

```bash
docker pull waticlems/prism_embedder:latest
docker run --rm -it \
    -v </path/to/your/slide.tif>:/input/images/whole-slide-image/slide.tif:ro \
    -v </path/to/your/mask.tif>:/input/images/tissue-mask/mask.tif:ro \
    -v model:/opt/ml/model
    -v </path/to/your/output/folder>:/output \
    waticlems/prism_embedder:latest
```

Upon execution, the following files will be saved in `/output`:

- `image-neural-representation.json`: 1280-dimensional slide-level embedding (JSON)
- `whole-slide-tiling-thumbnail.png`: PNG image showing the tissue tiling layout

## 📌 TODO

- [ ] add tSNE plot
- [ ] improve CLI customization (e.g., for spacing, resolution)