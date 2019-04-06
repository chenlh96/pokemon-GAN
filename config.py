import torch

class configBase():
    PATH_MODEL = None
    IS_ADD_I2V_TAG = False
    BATCH_SIZE = 16
    DIM_IMG = 64
    DIM_NOISE = 100
    LEARNING_RATE = 0.0002
    MOMENTUM = 0.5
    EPOCHS = 1
    INIT = True
    DEVICE = torch.device("cpu")

class config_dcgan(configBase):
    PATH_MODEL = '../model/dcgan.pth'
    IS_ADD_I2V_TAG = False
    BATCH_SIZE = 16
    DIM_IMG = 64
    DIM_NOISE = 100
    LEARNING_RATE = 0.0002
    MOMENTUM = 0.5
    EPOCHS = 1
    INIT = True
    DEVICE = torch.device("cpu")

class config_illustration_gan(configBase):
    PATH_MODEL = '../model/illustration-gan.pth'
    IS_ADD_I2V_TAG = False
    BATCH_SIZE = 1
    DIM_IMG = 64
    DIM_NOISE = 100
    LEARNING_RATE = 0.0002
    MOMENTUM = 0.5
    EPOCHS = 1
    INIT = True
    DEVICE = torch.device("cpu")

