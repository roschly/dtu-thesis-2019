import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose, Reshape, Dropout, BatchNormalization, Lambda, InputLayer, TimeDistributed
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.regularizers import l2

import config
from utils import chain_layer_functions


K = config.K
DROPOUT_RATE = config.dropout_rate

IMG_DIM = config.img_dim
INPUT_DIM = IMG_DIM[:-1] + ( IMG_DIM[-1]*K ,)
input_shape = IMG_DIM

base_depth = config.learner_base_depth

learner_encoded_size = config.learner_encoded_size
meta_encoded_size = config.meta_encoded_size

l2_val = config.meta_kernel_reg_l2

meta_prior_beta = config.meta_prior_beta

learner_hidden_activation = config.learner_hidden_activation

meta_learner_hidden_activation = config.meta_learner_hidden_activation
meta_learner_output_activation = config.meta_learner_output_activation

