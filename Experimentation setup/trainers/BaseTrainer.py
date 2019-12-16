
import matplotlib.pyplot as plt

import config
import utils

class BaseTrainer:
    def __init__(self):
        self.EPOCHS = config.epochs
        self.BATCH_SIZE = config.batch_size
        self.BATCHES_IN_EPOCH = config.batches_in_epoch
        self.K = config.K
        self.NUM_ADAPTATIONS = config.num_adaptations
        self.ALPHA = config.alpha
        self.loss_fn = config.loss_fn
        self.IMG_DIM = config.img_dim


    def generate_new_samples(self, prior, decoder, sample_size=config.K):
        z = prior.sample(sample_size)
        image_dist_new = decoder(z)
        return utils.sample_images(image_dist_new)
    