from tensorflow import keras
from keras import layers

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_unet(input_shape=(480, 584, 3)):
    print("Started building a  model...")
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D(2)(c3)

    b = conv_block(p3, 256)

    u3 = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    u3 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = conv_block(u1, 32)

    out = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(u1)
    model = keras.models.Model(inputs, out, name="unet_sharpen")
    print("Finished building a  model...")
    return model
