from src_c.StyleGAN2.stylegan2_pytorch import StyleGan2Trainer



if __name__ == '__main__':
    model = StyleGan2Trainer(name='model8', 
                             models_dir='./my_data/Stylegan2_newstructure/', 
                             pre_generated_uv_textures_dir='my_data/uv_textures_128',
                             uv_textures_pregenerated=True)
    model.prepare_for_training()
    # model.load()
    model.train_model()
    # model.load()
    # model.visualize_data2('pngs/fake_model3', index=2)
    # model.clear()
    # model.train_model()
    # fid = model.compute_fid_score()

    # print(fid)