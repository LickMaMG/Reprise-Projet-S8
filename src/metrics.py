from keras.metrics import mean_squared_error
import numpy as np

class MSE:
    name="mse"
    def __call__(self, y_true, y_pred):
        return np.mean(mean_squared_error(y_true, y_pred))