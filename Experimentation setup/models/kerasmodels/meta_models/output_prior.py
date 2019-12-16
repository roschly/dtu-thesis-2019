

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



def _create_decoder_model():
    # hidden_act = "relu"
    hidden_act = tf.nn.leaky_relu
    # out_act = "tanh"
    out_act = "relu"

    decoder_input = Input(shape=(meta_encoded_size,), name="decoder_input")
    x = Dense(32, name="dec_1", activation=hidden_act, kernel_regularizer=l2(l2_val))(decoder_input)
    x = Dense(32, name="dec_2", activation=hidden_act, kernel_regularizer=l2(l2_val))(x)

    location = Dense(learner_encoded_size, activation=out_act)(x)
    # scale = Dense(prior_scale_size, activation=out_act)(x)
    # output_layers = [location, scale]

    decoder = Model(inputs=decoder_input, outputs=location, name="decoder")
    return decoder



def create_meta_models():
    encoder = _create_encoder_model()
    decoder = _create_decoder_model()

    # FULL MODEL
    full_model_input = Input(shape=INPUT_DIM, name="full_model_input")
    encoded_img = encoder(full_model_input)
    decoded_img = decoder(encoded_img)
    full_model = Model(inputs=full_model_input, outputs=decoded_img, name="full_model")
    
    return encoder, decoder, full_model






