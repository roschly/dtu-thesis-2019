
from models.kerasmodels.shared_imports import *

import tensorflow as tf


def create_learner_prior(location=tf.zeros(learner_encoded_size)):
    return tfd.MultivariateNormalTriL(loc=location)

# Pure Conv - no dense bottleneck - simple sequential models
def create_learner_models(deactivate_layers=False):
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
        # tfp.layers.MultivariateNormalTriL(
        #     learner_encoded_size,
        #     activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior)),
        tfp.layers.MultivariateNormalTriL(learner_encoded_size)
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


    return encoder, decoder, vae