from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import VGG19

def build_model(choice,input_shape=(480, 584, 3)):
    """
        Builds the proper model based on the choice parameter
    """
    print("Started building a  model...")
    if choice == 'vgg19':
        model =  build_vgg19(input_shape)
    elif choice == 'resnet':
        model = build_resunet(input_shape)
    elif choice == 'unet':
        model = build_unet(input_shape)
    else :
        raise Exception('Invalid choice of a model')
    print("Finished building a  model...")
    return model


def conv_block(x, filters):
    """
        Builds a convolutional block used by unet-like model
    """
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_unet(input_shape=(480, 584, 3)):
    """
        Builds a U-Net–like model with:
        - Encoder: conv + pooling layers to capture context
        - Bottleneck: deepest features
        - Decoder: upsampling + skip connections to recover details
    """
    inputs = layers.Input(shape=input_shape)

    #Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D(2)(c3)

    #Bottleneck
    b = conv_block(p3, 256)

    # Decoder
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
    return model


def residual_block(x, filters):
    """
            Builds a residual block used by unet-like model
    """
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x


def build_resunet(input_shape=(480, 584, 3)):
    """
        Builds a ResUNet-like model with:
        - Encoder: residual blocks + pooling to capture features
        - Bottleneck: deepest features
        - Decoder: upsampling + skip connections using residual blocks
        - Output: residual added to input - model only learns the changes it has to apply to the given picture
    """
    inputs = layers.Input(shape=input_shape)

    c1 = residual_block(inputs, 32)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = residual_block(p2, 128)
    p3 = layers.MaxPooling2D(2)(c3)

    b = residual_block(p3, 256)

    u3 = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    u3 = residual_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = residual_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = residual_block(u1, 32)

    out_residual = layers.Conv2D(3, 3, padding="same")(u1)
    out = layers.Add()([inputs, out_residual])
    out = layers.Activation("sigmoid")(out)

    model = keras.models.Model(inputs, out, name="resunet_sharpen")
    return model


# Transfer learning
def build_vgg19(input_shape=(480, 584, 3)):
    """
        Builds a VGG19-based encoder-decoder model with:
        - Encoder: VGG19 convolutional layers (trained from scratch)
        - Decoder: upsampling + conv layers to reconstruct image
    """
    input_img = layers.Input(shape=input_shape)

    vgg = VGG19(weights=None, include_top=False, input_tensor=input_img)
    for layer in vgg.layers:
        layer.trainable = True

    encoder_output = vgg.get_layer("block3_conv3").output

    x = layers.UpSampling2D((2, 2))(encoder_output)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
    model = keras.models.Model(inputs=input_img, outputs=decoded)
    return model

