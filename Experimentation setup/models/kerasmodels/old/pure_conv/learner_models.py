
from models.kerasmodels.shared_imports import *

import tensorflow as tf

# Pure Conv - no dense bottleneck - simple sequential models
def create_learner_models(deactivate_layers=False):
    prior_loc = tf.Variable(tf.zeros(learner_encoded_size), trainable=False)
    prior_scale = tf.Variable(1.0, trainable=False)

    prior = tfd.Independent(
        tfd.Normal(loc=prior_loc, scale=prior_scale),
        reinterpreted_batch_ndims=1
    )

    # prior = tfd.Independent(
    #     tfd.Normal(loc=tf.zeros(learner_encoded_size), scale=1),
    #     reinterpreted_batch_ndims=1
    # )
    
    # set trainable = False, to only update weights of trainable = True (default for all layers)
    # used for identifying which layers to apply weights to, in MetaModelSomeLayers
    train_status = not deactivate_layers

    encoder = Sequential([
        InputLayer(input_shape=input_shape),
        # Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        Conv2D(base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
        Conv2D(base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
        Conv2D(2 * base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
        Conv2D(2 * base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu, trainable=train_status),
        Conv2D(2 * base_depth + 6, 7, strides=4, padding='same', activation=tf.nn.leaky_relu, trainable=train_status),
        Flatten(),
    #     tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(learner_encoded_size), activation=None),
        tfp.layers.MultivariateNormalTriL(
            learner_encoded_size,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior)),
    ])

    decoder = Sequential([
        InputLayer(input_shape=[learner_encoded_size]),
        Reshape([1, 1, learner_encoded_size]),
        Conv2DTranspose(2 * base_depth, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu, trainable=train_status),    
        Conv2DTranspose(2 * base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu, trainable=train_status),
        Conv2DTranspose(2 * base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
        Conv2DTranspose(base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
        Conv2DTranspose(base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
        Conv2DTranspose(base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
        Conv2D(filters=1, kernel_size=5, strides=1, padding='same', activation=None),
        Flatten(),
        tfp.layers.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
    ])

    vae = Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))

    models_dict = {
        "prior": prior,
        "prior_location": prior_loc,
        "prior_scale": prior_scale,
        "encoder": encoder,
        "decoder": decoder,
        "full_model": vae
    }

    return models_dict