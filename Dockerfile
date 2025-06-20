ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Amsterdam
USER root

# create input/output directory
RUN mkdir -p /input /output && \
    chown -R user:user /input /output

# Set /home/user as working directory
WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev \
    zlib1g-dev \
    curl \
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose port for ssh and jupyter
EXPOSE 22 8888

# Install python
RUN apt-get update && apt-get install -y python3-pip python3-dev python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ASAP
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN apt-get update && curl -L ${ASAP_URL} -o /tmp/ASAP.deb && apt-get install --assume-yes /tmp/ASAP.deb && \
    echo "/opt/ASAP/bin" > /usr/lib/python3/dist-packages/asap.pth && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app/

# You can add any Python dependencies to requirements.in
RUN python -m pip install --upgrade pip setuptools pip-tools \
    && rm -rf /home/user/.cache/pip

# install requirements
COPY --chown=user:user . /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.in \
    && rm -rf /home/user/.cache/pip

# install prism_embedder baseline
COPY --chown=user:user prism_embedder /opt/app/prism_embedder
COPY --chown=user:user LICENSE /opt/app/prism_embedder/
COPY --chown=user:user setup.py /opt/app/prism_embedder/
COPY --chown=user:user setup.cfg /opt/app/prism_embedder/
COPY --chown=user:user README.md /opt/app/prism_embedder/
COPY --chown=user:user pyproject.toml /opt/app/prism_embedder/

RUN python -m pip install  /opt/app

# Switch to user
USER user

#### Configure entrypoint
ENTRYPOINT ["prism_embedder"]