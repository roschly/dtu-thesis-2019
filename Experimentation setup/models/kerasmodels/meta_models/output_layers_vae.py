

from models.kerasmodels.shared_imports import *



# def _create_encoder_model():
#     # hidden_act = "relu"
#     hidden_act = tf.nn.leaky_relu

#     prior = tfd.Independent(
#         tfd.Normal(loc=tf.zeros(meta_encoded_size), scale=1.0),
#         reinterpreted_batch_ndims=1
#     )

#     # ENCODER
#     encoder_input = Input(shape=INPUT_DIM, name="encoder_input")
#     # x = Dropout(DROPOUT_RATE)(encoder_input)
#     x = Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val) )(x)
#     # x = Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act)(encoder_input)
#     # x = BatchNormalization()(x)
#     x = Dropout(DROPOUT_RATE)(x)
#     x = Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
#     # x = BatchNormalization()(x)
#     x = Dropout(DROPOUT_RATE)(x)
#     x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
#     # x = BatchNormalization()(x)
#     x = Dropout(DROPOUT_RATE)(x)
#     x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
#     # x = BatchNormalization()(x)
#     x = Dropout(DROPOUT_RATE)(x)

#     x = Flatten()(x)
#     # x = Dense(2*meta_encoded_size, activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
#     x = Dense(tfp.layers.MultivariateNormalTriL.params_size(meta_encoded_size), activation=None)(x)
#     encoded = tfp.layers.MultivariateNormalTriL(
#             meta_encoded_size,
#             activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=meta_prior_beta))(x)

#     encoder = Model(inputs=encoder_input, outputs=encoded, name="encoder")
#     return encoder



        

def _create_encoder_model():
    # hidden_act = "relu"
    hidden_act = tf.nn.leaky_relu

    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(meta_encoded_size), scale=1.0),
        reinterpreted_batch_ndims=1
    )

    # ENCODER
    encoder = Sequential([
        TimeDistributed(
            Conv2D(16, 3, strides=2, padding='same', activation=tf.nn.leaky_relu),
            input_shape=(config.K, 28,28,1)
        ),
        TimeDistributed(
            Conv2D(16, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        ),
        TimeDistributed(
            Conv2D(32, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        ),
        TimeDistributed(
            Conv2D(64, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        ),
        tf.keras.layers.GlobalAvgPool3D(),
        Dense(tfp.layers.MultivariateNormalTriL.params_size(meta_encoded_size), activation=None),
        tfp.layers.MultivariateNormalTriL(
            meta_encoded_size,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=meta_prior_beta))
    ])

    # encoder = Sequential([
    #     Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val) ),
    #     Dropout(DROPOUT_RATE),
    #     Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val)),
    #     Dropout(DROPOUT_RATE),
    #     Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val)),
    #     Dropout(DROPOUT_RATE),
    #     Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val)),
    #     Dropout(DROPOUT_RATE),
    #     Flatten(),
    #     Dense(tfp.layers.MultivariateNormalTriL.params_size(meta_encoded_size), activation=None),
    #     tfp.layers.MultivariateNormalTriL(
    #         meta_encoded_size,
    #         activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=meta_prior_beta))
    # ])

    return encoder




# def _create_decoder_model(params_list):
#     hidden_act = "relu"
#     out_act = "tanh"

#     # DECODER
#     # decoder_input = Input(shape=(meta_encoded_size,), name="decoder_input")
#     # x = Dense(32, name="dec_1", activation=hidden_act, kernel_regularizer=l2(l2_val))(decoder_input)
#     # x = BatchNormalization()(x)
#     # x = Dropout(DROPOUT_RATE)(x)
#     # x = Dense(64, name="dec_2", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
#     # x = BatchNormalization()(x)
#     # x = Dropout(DROPOUT_RATE)(x)

#     decoder_input = Input(shape=(meta_encoded_size,), name="decoder_input")
#     x = Dense(32, name="dec_1", activation=hidden_act, kernel_regularizer=l2(l2_val))(decoder_input)
#     x = Dense(32, name="dec_2", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)

#     # create output layers from weights_list
#     output_layers = []
#     for i in range( len(params_list) // 2 ):
#         kernel_size = params_list[i*2]
#         bias_size = params_list[i*2+1]
#         k_dense = Dense(kernel_size, name=f"k_{i}", activation=out_act)(x)
#         b_dense = Dense(bias_size, name=f"b_{i}", activation=out_act)(x)
#         output_layers.append(k_dense)
#         output_layers.append(b_dense)
        
#     decoder = Model(inputs=decoder_input, outputs=output_layers, name="decoder")
#     return decoder

def _create_decoder_model(weights_list_encoder, weights_list_decoder):
    # hidden_act = "relu"
    hidden_act = tf.nn.leaky_relu
    out_act = "tanh"

    decoder_input = Input(shape=(meta_encoded_size,), name="decoder_input")
    x = Dense(32, name="dec_1", activation=hidden_act, kernel_regularizer=l2(l2_val))(decoder_input)
    x = Dense(32, name="dec_2", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)

    # create output layers from weights_list
    output_layers = []
    for weights_list in [weights_list_encoder, weights_list_decoder]:
        for i in range( len(weights_list) // 2 ):
            kernel_size = weights_list[i*2]
            bias_size = weights_list[i*2+1]
            # k_dense = Dense(kernel_size, name=f"k_{i}", activation=out_act)(x)
            # b_dense = Dense(bias_size, name=f"b_{i}", activation=out_act)(x)
            k_dense = Dense(kernel_size, activation=out_act)(x)
            b_dense = Dense(bias_size, activation=out_act)(x)
            output_layers.append(k_dense)
            output_layers.append(b_dense)
        
    decoder = Model(inputs=decoder_input, outputs=output_layers, name="decoder")
    return decoder









# def create_latent_model(params_list):
#     encoder = _create_encoder_model()
#     decoder = _create_decoder_model(params_list)

#     # FULL MODEL
#     full_model_input = Input(shape=INPUT_DIM, name="full_model_input")
#     encoded_img = encoder(full_model_input)
#     decoded_img = decoder(encoded_img)
#     full_model = Model(inputs=full_model_input, outputs=decoded_img, name="full_model")
    
#     return encoder, decoder, full_model

def create_meta_models(weights_list_encoder, weights_list_decoder):
    encoder = _create_encoder_model()
    decoder = _create_decoder_model(weights_list_encoder, weights_list_decoder)

    # FULL MODEL
    # print("INPUT_DIM", INPUT_DIM)
    INPUT_DIM = (10,28,28,1)
    full_model_input = Input(shape=INPUT_DIM, name="full_model_input")
    encoded_img = encoder(full_model_input)
    decoded_img = decoder(encoded_img)
    full_model = Model(inputs=full_model_input, outputs=decoded_img, name="full_model")
    
    return encoder, decoder, full_model






