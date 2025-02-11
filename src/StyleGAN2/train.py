import wandb

from src.StyleGAN2.stylegan2_pytorch import StyleGan2Trainer

from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()




if __name__ == '__main__':
    model = StyleGan2Trainer(name='model_without_conditional_input', 
                             models_dir='./my_data/StyleGAN2/', 
                             uv_textures_pregenerated=True,
                             conditional_input=False,
                             image_size=256,
                             texture_size=128
                             )

    # model.prepare_for_training()
    # model.train_model()
    model.load()
    model.init_dataset()
    for _ in range(20):
        data = model.visualize_data(888)
        wandb.log(data)

    # fid = model.compute_fid_score()

    # print(fid)
    #model11 high res without con
    #model 12 high res with con