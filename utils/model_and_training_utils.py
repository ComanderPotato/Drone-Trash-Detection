import keras
import tensorflow as tf

def res_block(x, filters, kernel_size: tuple = (3, 3), strides = 1, activation = 'relu'):
    shortcut = x
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation(activation)(x)

    return x


def build_unet(input_shape: tuple) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    num_filters = 64

    skip_connections = []
    x = inputs

    # Encoder
    for _ in range(4):
        for _ in range(3):
            x = keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same')(x)
            x = keras.layers.Activation('relu')(x)
        skip_connections.append(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        num_filters *= 2

    # Bottleneck
    for _ in range(3):
        x = keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding="same")(x)
        x = keras.layers.Activation('relu')(x)
    
    # Decoder
    for i in range(4):
        x = keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=2, strides=2, padding="same")(x)
        num_filters //= 2
        x = keras.layers.Concatenate()([x, skip_connections[-(i + 1)]])
        for _ in range(3):
            x = keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding="same")(x)
            x = keras.layers.Activation('relu')(x)
        
    outputs = keras.layers.Conv2DTranspose(filters=2, kernel_size=(2, 2))(x)

    return keras.Model(inputs, outputs)
    

build_unet((224, 224, 3)).summary()
        
