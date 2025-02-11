# Use a more recent Ubuntu version
FROM ubuntu:20.04

# Install dependencies for Blender and Conda
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        bzip2 \
        libfreetype6 \
        libgl1-mesa-dev \
        libglu1-mesa \
        libxi6 \
        libxrender1 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libx11-6 \
        xvfb \
        python3 \
        snapd \
        git \
        wget && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

# # Download and install Blender
ENV BLENDER_BZ2_URL=https://mirror.clarkson.edu/blender/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
RUN mkdir /usr/local/blender && \
    curl -SL "$BLENDER_BZ2_URL" -o blender.tar.xz && \
    tar -Jxvf blender.tar.xz -C /usr/local/blender --strip-components=1 && \
    rm blender.tar.xz

# Add Blender to PATH
ENV PATH="/usr/local/blender:${PATH}"

# Install Miniconda
ENV MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Set environment variables for Conda
ENV PATH="$CONDA_DIR/bin:$PATH"
RUN conda init bash


# Create a new conda environment and install PyTorch3D dependencies
RUN conda create -n pytorch3denv -y && \
    conda run -n pytorch3denv conda install -y \
    pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda run -n pytorch3denv conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda run -n pytorch3denv conda install -y -c bottler nvidiacub && \
    conda run -n pytorch3denv conda install -y matplotlib && \
    conda run -n pytorch3denv conda install -y -c conda-forge wandb && \
    conda run -n pytorch3denv conda install -y -c pytorch3d pytorch3d


# RUN conda run -n pytorch3denv pip install pytorch_fid kornia vector_quantize_pytorch einops aim

WORKDIR /app
ENV PYTHONPATH=/app






