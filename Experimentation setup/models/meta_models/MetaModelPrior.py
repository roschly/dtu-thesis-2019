

import numpy as np
import tensorflow as tf

import config
from models.meta_models.MetaModelBase import MetaModelBase

from models.kerasmodels.current_models import create_learner_models, create_learner_prior
from models.kerasmodels.meta_models.output_prior import create_meta_models



class MetaModelPrior(MetaModelBase):
    def __init__(self):
        super().__init__()
        # learner
        encoder, decoder, full_model = create_learner_models()
        self.learner_prior = create_learner_prior()
        self.learner_encoder = encoder
        self.learner_decoder = decoder
        self.learner_model = full_model

            
        # meta learner
        enc, dec, full_model = create_meta_models()
        self.meta_encoder = enc
        self.meta_decoder = dec
        self.meta_model = full_model # full model

        self.save_model_plots()


    def apply_weights_from_mapping(self, weights):
        self.learner_prior = create_learner_prior(weights)


    def meta_learning_step(self, gradient_tape, batch_loss):
        models = [
            self.meta_model,
            self.learner_model
        ]
        for model in models:
            grads = gradient_tape.gradient(batch_loss, model.trainable_variables)
            self.optimizer.apply_gradients( zip(grads, model.trainable_variables) )






