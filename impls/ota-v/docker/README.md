# OGbench Docker Setup

## Building the Docker Image

There are two options for preparing the Docker image.

### Option 1 — Pull from Docker Hub

The easiest way is to pull the prebuilt OGBench [image](https://hub.docker.com/r/chwoong/ogbench) from Docker Hub.
The OGbench Docker image is built with the following base and dependencies:
- Base Image: `nvidia/cuda:12.1.0-base-ubuntu22.04`
- OGBench Version: `1.1.5`
- JAX Version: `0.6.2`

```bash
docker pull chwoong/ogbench:1.1.5
```

### Option 2 — Build locally from Dockerfile

If you want to customize, build the Docker image locally.

```bash
bash ./docker_build.sh
```


## Running the Docker Image

Before running the container, you must provide your WandB API key.

```bash
export IMAGE_NAME="chwoong/ogbench"  # or "ogbench" if you build the image locally
export WANDB_API_KEY="your_wandb_api_key_here"
export HOME_DIR="/path/to/ogbench/home"  # OGbench data directory (for mounting)
bash ./docker_run.sh
```
