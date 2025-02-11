# Generating textures with a GAN


## Getting started

docker pull tobiasvavroch/generating-textures-image
docker run -it -v $(pwd):/app --gpus all tobiasvavroch/generating-textures-image /bin/bash
conda activate pytorch3denv







To clone the repository, run:
```
git clone git@gitlab.mff.cuni.cz:vavrocht/generating-textures-with-a-gan.git
```
Make sure conda is installed. Checkout https://engineeringfordatascience.com/posts/install_miniconda_from_the_command_line/. 
Create new conda environment and install pytorch3d library:
```
conda create -n pytorch3denv
conda activate pytorch3denv
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
conda install matplotlib
conda install -c conda-forge wandb
conda install pytorch3d -c pytorch3d
```

More information: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

The project is split into two parts: 
1. Texture prediction via differentiable rendering
2. Training a GAN to produce suitable textures

Make sure to execute all code from the generating-textures-with-a-gan directory.
Also note, that the project structure is fixed (TODO: give an option to store data where user wants).

I recommend to run the project on mayrau server where the 3DDataset is stored. 


## Texture prediction

To run grid search:
```
python src/texture_optimization/predict_texture.py
```

Hyperparameters can be eddited in src/texture_optimization/predict_texture.py in class TextureOptimizationGridSearch - variable self.param_grid.
After grid search is finished, 2 combinations of best parameters will be computed on the basis of two different metrics.

TODO: Use the parameters from grid search for texture optimization. (It is implemented in in predict_texture.ipynb, just needs to be rewritten to normal python).

## GAN training

To train a GAN:
```
python src/GAN/train_WGAN_renderer.py 
```

Both networks will be stored in my_data/GAN_model. 
If networks are present in this directory, they will be loaded. Otherwise new networks will be initialized.
Data (textures and views rendered with those textures) will be logged in wandb. 
