import json

#LATENT VECTOR
LATENT_VECTOR_SIZE = 100
RANDOM_PART_SIZE = 50
NON_RANDOM_PART_SIZE = LATENT_VECTOR_SIZE - RANDOM_PART_SIZE


#SIMILARITY LOSS
SIMILARITY_LOSS_FLAG = True
SIMILARITY_LOSS_NUMBER_OF_EXAMPLES = 3

#WGAN settings
WGAN_DISCRIMINATOR_WEIGHT_CLIPPING = (-0.01, 0.01)
DISCRIMINATOR_LR = 0.0002
ENCODER_LR = 0.0002
DECODER_LR = 0.0002


#DATA LOADING
NUMBER_OF_MESHES_PER_ITEERATION = 4
NUMBER_OF_MESH_VIEWS = 16
NUM_EPOCHS = 10
DATAROOT = "/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model"
UNDERENDERABLE_DATA = "./my_data/nerenderovatelne.txt"


IMAGE_SIZE = 128

MANUAL_SEED = 410





def get_config():
    clean_data = {}
    for key, value in globals().items():
        try:
            json.dumps(value)
            clean_data[key] = value
        except TypeError:
            continue
    return clean_data



