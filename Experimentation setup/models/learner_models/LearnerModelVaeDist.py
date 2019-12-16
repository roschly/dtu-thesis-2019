
from models.BaseModel import BaseModel
from models.kerasmodels.current_models import create_learner_models, create_learner_prior
import utils
import config

class LearnerModelVaeDist(BaseModel):
    def __init__(self):
        super().__init__()
        enc, dec, full_model = create_learner_models()
        self.learner_prior = create_learner_prior
        self.learner_encoder = enc
        self.learner_decoder = dec
        self.learner_model = full_model

        self.save_model_plots()


    def copy_learner(self):
        prior_copy = create_learner_prior()
        enc_copy, dec_copy, model_copy = create_learner_models()

        model_weights = self.learner_model.get_weights()
        assert model_weights is not None
        model_copy.set_weights(model_weights)
        return prior_copy, enc_copy, dec_copy, model_copy


    def save_model_plots(self):
        if config.learner_models_plotted:
            return None
        else:
            config.learner_models_plotted = True

        zipped = [
            (self.learner_encoder, "learner_encoder"),
            (self.learner_decoder, "learner_decoder"),
            (self.learner_model,   "learner_model")
        ]
        utils.save_model_plots(models_and_names=zipped)
        


    
