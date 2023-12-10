from tensorflow import keras
import tensorflow as tf
from keras.metrics import mean_squared_error
import numpy as np

class MSE:
    name="mse"
    def __call__(self, y_true, y_pred):
        return np.mean(mean_squared_error(y_true, y_pred))

class PeakSignalNoiseRatio(keras.metrics.Metric):
    def __init__(self, name="psnr", max_pixel=256,**kwargs):
        super(PeakSignalNoiseRatio, self).__init__(name, **kwargs)
        self.max_pixel = max_pixel
        self.mse       = self.add_weight("mse", initializer="zeros")
        self.total     = self.add_weight("total", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        tf.debugging.assert_shapes([
            (y_true, ("batch_size", "height", "width", "channels")),
            (y_pred, ("batch_size", "height", "width", "channels"))
        ])

        squared_error = tf.square(y_true-y_pred)
        mse = tf.reduce_mean(squared_error, axis=[1, 2, 3])
        self.mse.assign_add(tf.reduce_sum(mse))
        self.count.assign_add(tf.cast(tf.size(y_true)[0], dtype=tf.float32))
    
    def result(self):
        psnr = 10. * tf.math.log(tf.square(self.max_pixel / self.mse / self.total)) / tf.math.log(10.)

        self.mse.assign(0)
        self.total.assign(0)

        return psnr
