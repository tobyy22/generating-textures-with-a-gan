import wandb

from src.StyleGAN2.stylegan2_pytorch import StyleGan2Trainer

from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()




if __name__ == '__main__':
    model = StyleGan2Trainer(name='model_with_encoder', 
                             models_dir='./my_data/StyleGAN2/', 
                             pre_generated_uv_textures_dir='my_data/uv_textures_128',
                             uv_textures_pregenerated=True,
                             conditional_input=True,
                             image_size=256,
                             texture_size=128,
                             evaluate_every=10
                             )

    model.prepare_for_training()
    model.train_model()
