import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .loss import ssim_loss
from PIL import Image
import matplotlib.pyplot as plt

def prepare(img_pil):
    img_np = np.array(img_pil) / 255.0
    img_np = cv2.resize(img_np, (584, 480))
    img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
    batch_1 = tf.expand_dims(img_tf, axis=0)
    return batch_1

def get_model(path):
    model = keras.models.load_model(
        path,
        custom_objects={"ssim_loss": ssim_loss}
    )
    return model

def predict_batch(model, batch):
    predicted = model.predict(batch)
    return predicted

def inference_one(img_pil,model_path):
    model = get_model(model_path)
    predicted = predict_batch(model, prepare(img_pil))
    return tf.squeeze(predicted, axis=0)


if __name__ == '__main__':
    img = Image.open("test/test3.jpg").convert("RGB")
    new = inference_one(img, "trained/model_latest.keras")
    plt.imshow(img)
    plt.show()
    plt.imshow(new)
    plt.show()