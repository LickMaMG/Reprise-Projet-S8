import tensorflow as tf
import numpy as np

class MSE:
    name="mse"
    def __call__(self, y_true, y_pred):
        return np.mean(tf.keras.metrics.mean_squared_error(y_true, y_pred))

class PeakSignalNoiseRatio(tf.keras.metrics.Metric):
    def __init__(self, name="psnr", max_pixel=1., **kwargs):
        super(PeakSignalNoiseRatio, self).__init__(name=name, **kwargs)
        self.max_pixel = tf.constant(max_pixel, dtype=tf.float32)
        self.mse = self.add_weight("mse", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):

        squared_error = tf.square(y_true - y_pred)
        mse = tf.reduce_mean(squared_error, axis=[1, 2, 3])
        self.mse.assign_add(tf.reduce_sum(mse))
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.total.assign_add(batch_size)
    
    def result(self):
        psnr = 10. * tf.math.log(self.max_pixel**2 / self.mse) / tf.math.log(10.) / self.total

        self.mse.assign(0)
        self.total.assign(0)

        return psnr

    # @property
    # def variables(self):
    #     # Return the variables as a list
    #     return [self.mse, self.total]

    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, "max_pixel": self.max_pixel}

    # @classmethod
    # def from_config(cls, config):
    #     sublayer_config = config.pop("max_pixel")
    #     sublayer = tf.keras.saving.deserialize_tf.keras_object(sublayer_config)
    #     return cls(sublayer, **config)
