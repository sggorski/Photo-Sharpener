import tensorflow as tf
from load import load_img

def get_dataset(image_paths, batch_size = 4) -> tf.data.Dataset:
    print("Started creating a  dataset...")
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    print("Finished creating a  dataset...")
    return ds