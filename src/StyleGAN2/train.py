import wandb

from src.StyleGAN2.stylegan2_pytorch import StyleGan2Trainer

from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()




if __name__ == '__main__':
    model = StyleGan2Trainer(name='model_without_encoder', 
                             models_dir='./my_data/StyleGAN2/', 
                             uv_textures_pregenerated=True,
                             conditional_input=False,
                             image_size=256,
                             texture_size=128,
                             dataset_path='/app/3Dataset'
                             )

    model.prepare_for_training()
    model.train_model()
