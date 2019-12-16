
from models.kerasmodels.shared_imports import *

# def create_learner_prior():
#     prior = tfd.Independent(
#         tfd.Normal(loc=tf.zeros(learner_encoded_size), scale=1),
#         reinterpreted_batch_ndims=1
#     )
#     return prior

def create_learner_prior():
    prior = tfd.MultivariateNormalTriL(loc=tf.zeros(learner_encoded_size))
    return prior

    

# Dense bottleneck - simple sequential models
def create_learner_models(deactivate_layers=False):
    prior = create_learner_prior()
    
    # set trainable = False, to only update weights of trainable = True (default for all layers)
    # used for identifying which layers to apply weights to, in MetaModelSomeLayers
    train_status = not deactivate_layers
    

    encoder = Sequential([
        InputLayer(input_shape=input_shape),
        # Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        Conv2D(base_depth, 5, strides=1, padding='same', activation=learner_hidden_activation),
        Conv2D(base_depth, 5, strides=2, padding='same', activation=learner_hidden_activation),
        Conv2D(2 * base_depth, 5, strides=1, padding='same', activation=learner_hidden_activation),
        Conv2D(2 * base_depth, 5, strides=2, padding='same', activation=learner_hidden_activation, trainable=train_status),
        Flatten(),
        Dense(tfp.layers.MultivariateNormalTriL.params_size(learner_encoded_size), activation=None, trainable=train_status),
        tfp.layers.MultivariateNormalTriL(
            learner_encoded_size,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=meta_prior_beta))
    ])

    decoder = Sequential([
        InputLayer(input_shape=[learner_encoded_size]),
        Reshape([1, 1, learner_encoded_size]),
        Conv2DTranspose(2 * base_depth, 7, strides=1, padding='valid', activation=learner_hidden_activation, trainable=train_status),    
        Conv2DTranspose(2 * base_depth, 5, strides=2, padding='same', activation=learner_hidden_activation, trainable=train_status),
        # Conv2DTranspose(2 * base_depth, 5, strides=1, padding='same', activation=learner_hidden_activation, trainable=train_status), #
        # Conv2DTranspose(2 * base_depth, 5, strides=2, padding='same', activation=learner_hidden_activation),#
        Conv2DTranspose(base_depth, 5, strides=1, padding='same', activation=learner_hidden_activation),
        Conv2DTranspose(base_depth, 5, strides=2, padding='same', activation=learner_hidden_activation),
        # Conv2DTranspose(base_depth, 5, strides=1, padding='same', activation=learner_hidden_activation), #
        Conv2D(filters=1, kernel_size=5, strides=1, padding='same', activation=None),
        Flatten(),
        # Dense(IMG_DIM[0] * IMG_DIM[1] * IMG_DIM[2], )
        tfp.layers.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
        # tfp.layers.IndependentBernoulli(input_shape),
    ])

    vae = Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))


    return encoder, decoder, vae