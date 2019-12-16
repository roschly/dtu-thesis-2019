
import abc

import tensorflow as tf

from models.BaseModel import BaseModel
import config
import utils

class MetaModelBase(abc.ABC, BaseModel):
    def __init__(self):
        super().__init__()


    # KL implicit in loss
    # def compute_loss(self, images, weights):
    #     # assert len(self.learner_model.trainable_variables) != 0, "compute loss"
    #     self.apply_weights_from_mapping(weights)
    #     # assert len(self.learner_model.trainable_variables) != 0, "compute loss"
    #     images_dist = self.learner_model(images)
    #     loss = self.loss_fn(images, images_dist)
    #     return loss

    # KL explicit in loss
    def compute_loss(self, images, weights):
        self.apply_weights_from_mapping(weights)
        images_dist = self.learner_model(images)
        recon_loss = self.loss_fn(images, images_dist)
        learner_z_dist = self.learner_encoder(images)
        kl_loss = self.learner_prior.kl_divergence(learner_z_dist)
        return recon_loss + config.meta_prior_beta*kl_loss

    #@abc.abstractmethod
    def compute_meta_loss(self, images, weights, z, z_):
        loss = self.compute_loss(images, weights)
        # extra loss, on the z encodings
        z_loss = tf.reduce_mean(tf.math.squared_difference(z, tf.stop_gradient(z_))) * self.GAMMA
        return loss + z_loss

    @abc.abstractmethod
    def meta_learning_step(self):
        pass

    def save_model_plots(self):
        if not config.learner_models_plotted:
            learner_models_and_names = [
                (self.learner_encoder, "learner_encoder"),
                (self.learner_decoder, "learner_decoder"),
                (self.learner_model, "learner_model")
            ]
            utils.save_model_plots(models_and_names=learner_models_and_names)
            config.learner_models_plotted = True

        if not config.meta_models_plotted:
            meta_models_and_names = [
                (self.meta_encoder, "meta_encoder"),
                (self.meta_decoder, "meta_decoder"),
                (self.meta_model, "meta_model")
            ]
            utils.save_model_plots(models_and_names=meta_models_and_names)
            config.meta_models_plotted = True


        