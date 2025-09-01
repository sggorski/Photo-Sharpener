import tensorflow as tf

def ssim_loss(y_true, y_pred):
    """
        Structural similarity index measure used in training along with  MAE
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))