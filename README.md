# Project Setup and Experiments

## Registering on Weights & Biases (wandb)
To log experimental data, create an account on the platform [Weights & Biases](https://wandb.ai/).
In your user settings, you will find an API key. Store it securely.

## Source Code
Clone the project using the following command:

```bash
git clone --branch main --depth 1 \
    git@github.com:tobyy22/generating-textures-with-a-gan.git
```

## Dataset and Test Data
The dataset and test data are available on the MFF UK server. They will be accessible in the environment once mounted into the Docker container (see below).
Eventually the data can be downloaded through https://drive.google.com/file/d/1CIpys_H4ZhknnZp_9W6r5tpHDhRLCkVV/view?usp=share_link (without dataset). Unzip and paste into directory with the repository. 

## Environment Setup
The entire project runs in a Docker environment. First, you need to build a new image from the `Dockerfile` in the project repository using the following command (though a pre-built image is available in the next step):

```bash
docker build -t generating-textures-image .
```

To simplify the process, a pre-built image is available on Docker Hub and can be downloaded with:

```bash
docker pull tobiasvavroch/generating-textures-image
```

The downloaded image contains only the prepared environment, not the source code. The source code can be mounted into the container when it is run.

## Running the Environment
Run the environment with GPU support and mount the current directory to `/app`:

```bash
docker run -it -v $(pwd):/app \
   -v /projects/3DDatasets/3D-FUTURE/3D-FUTURE-model:/app/3Dataset \
   -v /projects/tobiasvavroch_bc_data:/app/my_data  --gpus all \
   tobiasvavroch/generating-textures-image /bin/bash
conda activate pytorch3denv
wandb login
```

You will need to log in using your API key.
Once the environment is activated, you can run experiments. Each experiment automatically creates a new run in `wandb`, where the logged results can be monitored.

## Running Experiments
Now everything is set up to run experiments.

### Experiment with Texture Optimization [Section Reference: Experiment Gradient Rasterizer]

### Cow Model Optimization
```bash
python3 src/texture_optimization/optimize_cow.py
```

### Grid Search for Optimization Parameters
```bash
python3 src/texture_optimization/run_grid_search.py
```

### Overfitting Experiment with GAN and WGAN [Section Reference: Overfitting GAN]
```bash
python3 src/DCWGAN/overfit_gan.py
```

To switch between GAN and WGAN, set the `wgan` parameter to `True/False`.

### WGAN + Rasterizer Experiment [Section Reference: WGAN Rasterizer]

**Training:**
```bash
python3 src/DCWGAN/train_dcwgan_renderer.py
```

**Training with Higher Resolution:**
```bash
python3 src/DCWGAN/train_dcwgan_renderer_higher_resolution.py
```

**Visualizing Results:**
```bash
python3 src/evaluate_models.py --trainer DCWGANRenderer \
    --visualize_results --visualized_object_id 1433
```

**Computing FID Score (runs for a long time):**
```bash
python3 src/evaluate_models.py \
    --trainer DCWGANRenderer \
    --evaluate_fid_score
```

### U-Net + Rasterizer Experiment [Section Reference: U-Net Rasterizer]

**Training:**
```bash
python3 src/PytorchMRIUnet/train.py
```

**Training with Similarity Loss:**
```bash
python3 src/PytorchMRIUnet/train_with_similarity_loss.py
```

**Visualizing Results:**
```bash
python3 src/evaluate_models.py --trainer PytorchUnet \
    --visualize_results --visualized_object_id 1433
```

**Computing FID Score (runs for a long time):**
```bash
python3 src/evaluate_models.py \
    --trainer PytorchUnet \
    --evaluate_fid_score
```

For the similarity loss experiment, select `PytorchUnetSimilarityLoss` as the `trainer`.

To reproduce the experiment with a custom U-Net, use the `WGANUnet` directory similarly.

### UV Unwrapping Using Blender
The following command performs UV unwrapping for a sample object file without a texture. It creates a black texture as a placeholder so that PyTorch3D can correctly load the model. The resulting directory can then be used as a single-element dataset for evaluation.

```bash
blender --background --python src/uv_unwrap.py -- \
    --obj_path my_data/octopus_object/octopus.obj \
    --export_dir ./octopus_unwrapped
```

You can then use the previous scripts with this dataset, for example:

```bash
python3 src/evaluate_models.py --trainer PytorchUnet \
    --visualize_results --visualized_object_id 0 \
    --dataset_path ./octopus_unwrapped
```

Results will again be available in `wandb`. 

