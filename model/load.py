import tensorflow as tf
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def get_image_paths(path):
    """
        Gets a list of image paths.
    """
    print("Started loading paths...")
    paths =  glob.glob(path)
    print("Finished loading paths...")
    return paths

def degrade_image(original):
    """
        Prepares image for the training, returns a pair: degraded image and original image.
        Applies randomised degeneration of an image, image can be left without any changes,
        so the model can also learn when not to apply any modifications
    """
    target = original.copy()
    degraded = original.copy()

    if np.random.rand() < 0.95:
        k = np.random.choice([3, 5, 7, 9, 15])
        sigma = np.random.uniform(0.5, 5.0)
        degraded = cv2.GaussianBlur(degraded, (k, k), sigmaX=sigma)

    if np.random.rand() < 0.9:
        k1 = np.random.choice([3, 5, 7])
        sigma1 = np.random.uniform(0.5, 1.0)
        k2 = np.random.choice([9, 15, 17])
        sigma2 = np.random.uniform(1.0, 5.0)
        mask = np.random.rand(*degraded.shape[:2])
        blur_light = cv2.GaussianBlur(degraded, (k1, k1), sigma1)
        blur_heavy = cv2.GaussianBlur(degraded, (k2, k2), sigma2)
        degraded = np.where(mask[..., None] > 0.5, blur_light, blur_heavy)

    return degraded, target


def _load_img(path):
    """
    Loads an image from the path and applies degrade_image function.
    """
    path = path.numpy().decode('utf-8')
    img = cv2.imread(path)
    img = cv2.resize(img, (584, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
    blurred,img = degrade_image(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    blurred = tf.convert_to_tensor(blurred, dtype=tf.float32)
    return blurred,img


def load_img(path):
  """
    Loads an image from the path and applies degrade_image function, returns tf tensors
  """
  blurred,img =  tf.py_function(func=_load_img, inp=[path], Tout=(tf.float32, tf.float32))
  blurred.set_shape([480, 584,3])
  img.set_shape([480, 584,3])
  return blurred,img

if __name__ == '__main__':
    PATH = r"C:\Users\sggor\Personal\photo_sharpener\images\test2017\*.jpg"
    a, b = load_img(get_image_paths(PATH)[0])
    plt.imshow(a)
    plt.show()
    plt.imshow(b)
    plt.show()
