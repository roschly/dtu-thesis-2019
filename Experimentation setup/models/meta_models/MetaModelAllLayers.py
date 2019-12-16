

import numpy as np
import tensorflow as tf

import config
from models.meta_models.MetaModelBase import MetaModelBase
from models.kerasmodels.current_models import create_learner_models, create_meta_models, create_learner_prior
# from models.kerasmodels.current_models import create_learner_models, create_meta_models


class MetaModelAllLayers(MetaModelBase):
    def __init__(self):
        super().__init__()
        # learner
        enc, dec, full_model = create_learner_models()
        self.learner_prior = create_learner_prior()
        self.learner_encoder = enc
        self.learner_decoder = dec
        self.learner_model = full_model # full model
        
        
        # learner ENCODER - weights mapping + list
        mapping, w_list = self.weights_mapping_and_list(model_type="encoder")
        self.learner_encoder_weights_mapping = mapping
        self.learner_encoder_weights_list = w_list
        
        # learner DECODER - weights mapping + list
        mapping, w_list = self.weights_mapping_and_list(model_type="decoder")
        self.learner_decoder_weights_mapping = mapping
        self.learner_decoder_weights_list = w_list

         
        # meta learner
        enc, dec, full_model = create_meta_models(
            weights_list_encoder=self.learner_encoder_weights_list, 
            weights_list_decoder=self.learner_decoder_weights_list
        )
        self.meta_encoder = enc
        self.meta_decoder = dec
        self.meta_model = full_model # full model

        self.save_model_plots()
   

    def weights_mapping_and_list(self, model_type):
        weights_mapping = []
        weights_list = []
        if model_type == "encoder":
            model = self.learner_encoder
        elif model_type == "decoder":
            model = self.learner_decoder
        for i, layer in enumerate(model.layers):
            # if "kernel" in dir(layer):
            if hasattr(layer, "kernel"):
                weights_mapping.append(i)
                kernel_shape = layer.kernel.shape
                bias_shape = layer.bias.shape
                # print(kernel_shape, "---", bias_shape)
                total_kernel_params = np.product(kernel_shape)
                total_bias_params = np.product(bias_shape)

                weights_list.append(total_kernel_params)
                weights_list.append(total_bias_params)
        return weights_mapping, weights_list


    # def compute_loss(self, images, weights):
    #     self.apply_weights_from_mapping(weights)
    #     images_dist = self.learner_model(images)
    #     loss = self.loss_fn(images, images_dist)
    #     # print(loss)
    #     return loss


    # def compute_meta_loss(self, images, weights, z, z_):
    #     loss = self.compute_loss(images, weights)
    #     # extra loss, on the z encodings
    #     z_loss = tf.reduce_mean(tf.math.squared_difference(z, tf.stop_gradient(z_))) * self.GAMMA
    #     return loss + z_loss


    def meta_learning_step(self, gradient_tape, batch_loss):
        grads = gradient_tape.gradient(batch_loss, self.meta_model.trainable_variables)
        # maxx = np.max( [np.max(g) for g in grads] )
        # print(maxx)
        # grads_clipped, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clipnorm_outer)
        # print(grads_clipped[0])
        # raise ValueError("!")
        # maxx = np.max( [np.max(g) for g in grads_clipped] )
        # print(maxx)
        
        # self.optimizer.apply_gradients( zip(grads_clipped, self.meta_model.trainable_variables) )
        self.optimizer.apply_gradients( zip(grads, self.meta_model.trainable_variables) )


    def apply_weights_from_mapping(self, weights):
        j = 0
        for i in self.learner_encoder_weights_mapping:
            kernel_shape = self.learner_encoder.layers[i].kernel.shape
            self.learner_encoder.layers[i].kernel = tf.reshape(weights[j], kernel_shape)
            bias_shape = self.learner_encoder.layers[i].bias.shape
            self.learner_encoder.layers[i].bias = tf.reshape(weights[j+1], bias_shape)
            j += 2

        for i in self.learner_decoder_weights_mapping:
            kernel_shape = self.learner_decoder.layers[i].kernel.shape
            self.learner_decoder.layers[i].kernel = tf.reshape(weights[j], kernel_shape)
            bias_shape = self.learner_decoder.layers[i].bias.shape
            self.learner_decoder.layers[i].bias = tf.reshape(weights[j+1], bias_shape)
            j += 2


    