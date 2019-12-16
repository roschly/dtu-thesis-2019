

import numpy as np
import tensorflow as tf

import config
from models.meta_models.MetaModelBase import MetaModelBase
from models.kerasmodels.current_models import create_learner_models, create_learner_prior


class MetaModelMaml(MetaModelBase):
    def __init__(self):
        super().__init__()
        # learner
        enc, dec, full_model = create_learner_models()
        self.learner_prior = create_learner_prior()
        self.learner_encoder = enc
        self.learner_decoder = dec
        self.learner_model = full_model # full model

        # print("!!!!!!!!!!!!!!!!!!!!")
        # print(self.learner_model.trainable_variables)
        # print("!!!!!!!!!!!!!!!!!!!!")
        
        
        # learner ENCODER - weights mapping + list
        mapping, w_list = self.weights_mapping_and_list(model_type="encoder")
        self.learner_encoder_weights_mapping = mapping
        self.learner_encoder_weights_list = w_list
        
        # learner DECODER - weights mapping + list
        mapping, w_list = self.weights_mapping_and_list(model_type="decoder")
        self.learner_decoder_weights_mapping = mapping
        self.learner_decoder_weights_list = w_list

         
        # self.save_model_plots()
   

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



    def meta_learning_step(self, gradient_tape, batch_loss):
        grads = gradient_tape.gradient(batch_loss, self.learner_model.trainable_variables)
        self.optimizer.apply_gradients( zip(grads, self.learner_model.trainable_variables) )


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


    def apply_gradients(self, gradients):
        print( type(gradients) )
        print(len(gradients))
        print(gradients.shape)
        j = 0
        for i in self.learner_encoder_weights_mapping:
            kernel_shape = self.learner_encoder.layers[i].kernel.shape
            self.learner_encoder.layers[i].kernel -= config.alpha * tf.reshape(gradients[j], kernel_shape)
            bias_shape = self.learner_encoder.layers[i].bias.shape
            self.learner_encoder.layers[i].bias -= config.alpha * tf.reshape(gradients[j+1], bias_shape)
            j += 2

        for i in self.learner_decoder_weights_mapping:
            kernel_shape = self.learner_decoder.layers[i].kernel.shape
            self.learner_decoder.layers[i].kernel -= config.alpha * tf.reshape(gradients[j], kernel_shape)
            bias_shape = self.learner_decoder.layers[i].bias.shape
            self.learner_decoder.layers[i].bias -= config.alpha * tf.reshape(gradients[j+1], bias_shape)
            j += 2


    def get_weights(self):
        weights = []
        for i in self.learner_encoder_weights_mapping:
            kernel = self.learner_encoder.layers[i].kernel
            bias = self.learner_encoder.layers[i].bias
            weights.append(kernel)
            weights.append(bias)

        for i in self.learner_decoder_weights_mapping:
            kernel = self.learner_decoder.layers[i].kernel
            bias = self.learner_decoder.layers[i].bias
            weights.append(kernel)
            weights.append(bias)

        return weights

        
    def set_weights(self, weights):
        self.apply_weights_from_mapping(weights)


    