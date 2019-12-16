

from models.kerasmodels.shared_imports import *


def _create_encoder_model():
    # hidden_act = "relu"
    hidden_act = tf.nn.leaky_relu

    # ENCODER
    encoder_input = Input(shape=INPUT_DIM, name="encoder_input")
    x = Dropout(DROPOUT_RATE)(encoder_input)
    x = Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val) )(x)
    # x = Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act)(encoder_input)
    # x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
    # x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
    # x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
    # x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Flatten()(x)
    x = Dense(2*meta_encoded_size, activation=hidden_act, kernel_regularizer=l2(l2_val))(x)
    encoded = Dense(meta_encoded_size, name="encoded", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)

    encoder = Model(inputs=encoder_input, outputs=encoded, name="encoder")
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
    full_model_input = Input(shape=INPUT_DIM, name="full_model_input")
    encoded_img = encoder(full_model_input)
    decoded_img = decoder(encoded_img)
    full_model = Model(inputs=full_model_input, outputs=decoded_img, name="full_model")
    
    return encoder, decoder, full_model






