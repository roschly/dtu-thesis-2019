""" 
    This module is used for switching between models when prototyping/experimenting 
"""

# DENSE
# from models.kerasmodels.dense_bottleneck.learner_models import create_learner_models
# from models.kerasmodels.dense_bottleneck.meta_models import create_meta_models

# PURE CONV 
# from models.kerasmodels.pure_conv.learner_models import create_learner_models
# from models.kerasmodels.pure_conv.meta_models import create_meta_models


# DENSE learner
# from models.kerasmodels.dense_bottleneck.learner_models import create_learner_models
# from models.kerasmodels.pure_conv.meta_models import create_meta_models




# LEARNER
from models.kerasmodels.learner_models.dense_bottleneck import create_learner_models, create_learner_prior
# from models.kerasmodels.learner_models.pure_conv_kl_reg import create_learner_models, create_learner_prior
# from models.kerasmodels.learner_models.pure_conv_no_kl import create_learner_models, create_learner_prior

# META
from models.kerasmodels.meta_models.output_layers import create_meta_models
# from models.kerasmodels.meta_models.output_layers_vae import create_meta_models
