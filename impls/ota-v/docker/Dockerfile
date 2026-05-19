# Use the official NVIDIA CUDA 12.1 base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install dependencies (Python, pip, etc.) and missing graphics libraries
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    curl \
    build-essential \
    libffi-dev \
    libegl1 \
    libglvnd-dev \
    libgles2 \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    gcc-12 \
    g++-12 \
    vulkan-tools \
    mesa-vulkan-drivers \
    pigz \
    git \
    git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Install PyOpenGL and PyOpenGL_accelerate
RUN pip install PyOpenGL PyOpenGL_accelerate

# Install JAX and the CUDA-enabled version of jaxlib
RUN pip install --no-cache-dir \
    jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify installation of JAX and CUDA-enabled version
RUN python -c "import jax; print(jax.__version__); import jaxlib; print(jaxlib.__version__);"

# Install ogbench dependencies
RUN pip install ogbench==1.1.5
RUN pip install --upgrade flax>=0.8.4 distrax>=0.1.5 ml_collections matplotlib moviepy==1.0.3 wandb opencv-python


# Default command to run on container start
CMD ["bash"]
