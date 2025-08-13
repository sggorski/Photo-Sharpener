import tensorflow as tf
import cv2
import glob

def get_image_paths(path):
    print("Started loading paths...")
    paths =  glob.glob(path)
    print("Finished loading paths...")
    return paths


def _load_img(path):
    path = path.numpy().decode('utf-8')
    img = cv2.imread(path)
    img = cv2.resize(img, (584, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    #noise = np.random.normal(0, 0.1, img.shape)
    #noisy = np.clip(img + noise, 0, 1)
    blurred = cv2.GaussianBlur(img, (9,9),cv2.BORDER_DEFAULT)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    blurred = tf.convert_to_tensor(blurred, dtype=tf.float32)
    return blurred,img


def load_img(path):
  blurred,img =  tf.py_function(func=_load_img, inp=[path], Tout=(tf.float32, tf.float32))
  blurred.set_shape([480, 584,3])
  img.set_shape([480, 584,3])
  return blurred,img