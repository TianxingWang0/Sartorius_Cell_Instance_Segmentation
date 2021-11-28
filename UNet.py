"""
UNet
"""

from tensorflow.keras import layers
import tensorflow.keras as keras


def get_UNet_Xception_model(img_size):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    previous_block_activation = layers.Activation("relu")(x)  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = layers.Activation("relu")(x)  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = layers.Activation("relu")(x)  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def get_light_UNet_transformer(img_size, weight_path=None):
    inputs = keras.Input(shape=img_size + (1,))

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # level one down block
    residual = layers.SeparableConv2D(32, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    level_1_corp = layers.Activation("relu")(x)

    # level two down block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_1_corp)

    residual = layers.SeparableConv2D(64, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    level_2_corp = layers.Activation("relu")(x)

    # level three down block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_2_corp)

    residual = layers.SeparableConv2D(128, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    level_3_corp = layers.Activation("relu")(x)

    # bottom block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_3_corp)

    residual = layers.SeparableConv2D(256, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 3 up block
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(level_3_corp, x)

    residual = layers.SeparableConv2D(64, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 2 up block
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(level_2_corp, x)

    residual = layers.SeparableConv2D(32, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 1 up block
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.concatenate([level_1_corp, x], axis=3)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    residual = layers.SeparableConv2D(32, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # output layer
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs)
    if weight_path is not None:
        model.load_weights(weight_path)
    return model


def get_light_UNet_transformer_no_conv(img_size):
    inputs = keras.Input(shape=img_size + (1,))

    # level one down block
    residual = layers.SeparableConv2D(32, 3, padding="same")(inputs)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    level_1_corp = layers.Activation("relu")(x)

    # level two down block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_1_corp)

    residual = layers.SeparableConv2D(64, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    level_2_corp = layers.Activation("relu")(x)

    # level three down block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_2_corp)

    residual = layers.SeparableConv2D(128, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    level_3_corp = layers.Activation("relu")(x)

    # bottom block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_3_corp)

    residual = layers.SeparableConv2D(256, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 3 up block
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(level_3_corp, x)

    residual = layers.SeparableConv2D(64, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 2 up block
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(level_2_corp, x)

    residual = layers.SeparableConv2D(32, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 1 up block
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.concatenate([level_1_corp, x], axis=3)

    residual = layers.SeparableConv2D(32, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # output layer
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


def get_UNet_transformer(img_size):
    inputs = keras.Input(shape=img_size + (1,))

    # level one down block
    residual = layers.SeparableConv2D(32, 3, padding="same")(inputs)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    level_1_corp = layers.Activation("relu")(x)

    # level two down block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_1_corp)

    residual = layers.SeparableConv2D(64, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)
    level_2_corp = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)

    # level three down block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_2_corp)

    residual = layers.SeparableConv2D(128, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)
    level_3_corp = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)

    # bottom block
    x = layers.MaxPooling2D(3, strides=2, padding="same")(level_3_corp)

    residual = layers.SeparableConv2D(256, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)

    # level 3 up block
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(level_3_corp, x)

    residual = layers.SeparableConv2D(64, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 2 up block
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(level_2_corp, x)

    residual = layers.SeparableConv2D(32, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # level 1 up block
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.concatenate([level_1_corp, x], axis=3)

    residual = layers.SeparableConv2D(32, 3, padding="same")(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])
    x = layers.Activation("relu")(x)

    # output layer
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
# keras.backend.clear_session()
