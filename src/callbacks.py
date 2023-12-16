import sys, io, math
sys.path.append("./")
sys.path.append("../")


import tensorflow as tf
from typing import List
from matplotlib.figure import Figure
import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class ExpSchedule(tf.keras.callbacks.LearningRateScheduler):
    "??? description"

    def __init__(self, lr0, s):
        self.lr0 = lr0
        self.s   = s
        super().__init__(schedule=self.expo_decay_fn)

    def expo_decay_fn(self, epoch):
        return self.lr0*0.1**(epoch/self.s)


class CosineSchedule(tf.keras.callbacks.LearningRateScheduler):
    "??? description"

    def __init__(self, eta, n_max):
        self.eta   = eta
        self.n_max = n_max
        super().__init__(schedule=self.cosine_decay_fn)

    def cosine_decay_fn(self, epoch):
        return self.eta*0.5*(1 + np.cos(epoch*np.pi/self.n_max))

class WarmupCosineSchedule(tf.keras.callbacks.LearningRateScheduler):
    "??? description"

    def __init__(self,
        base_lr,
        warmup_start_lr,
        warmup_epochs,
        max_epochs):
        
        self.base_lr   = base_lr
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs   = warmup_epochs

        super().__init__(schedule=self.decay_fn)
    
    def cosine_fn(self, epoch):
        return self.base_lr*0.5*(1+np.cos(epoch*np.pi/self.max_epochs))

    def decay_fn(self, epoch):
        lr = self.cosine_fn(epoch)
        
        if epoch  < self.warmup_epochs:
            lr_start = self.warmup_start_lr
            alpha = (self.cosine_fn(self.warmup_epochs) - lr_start) / self.warmup_epochs
            lr = epoch*alpha + lr_start
        return lr

class SaveDenoised(keras.callbacks.Callback):
    def __init__(self, logdir: str, val_generator) -> None:

        super().__init__()
        self.imgdir = os.path.join(logdir, "imgs")
        self.val_generator = val_generator
        self.batch_size = val_generator.batch_size
        self.writer = tf.summary.create_file_writer(logdir)

        os.makedirs(self.imgdir, exist_ok=True)


    def save_image(self, noised, denoised, save_as: str):
        rows = math.ceil(self.batch_size / 2)
        # cols = 2
        fig, axes = plt.subplots(rows, rows*2, figsize=(8,6))
        axes = axes.ravel()
        for i in range(self.batch_size):
            axes[i*2].imshow(noised[i], cmap="gray"); axes[i*2].set_xticks([]); axes[i*2].set_yticks([]); axes[i*2].set_title("Noised")
            axes[i*2+1].imshow(denoised[i], cmap="gray"); axes[i*2+1].set_xticks([]); axes[i*2+1].set_yticks([]); axes[i*2+1].set_title("Denoised")
        
        return self.figure_to_tf_image(fig, save_to=f"{self.imgdir}/{save_as}.png")
    
    def on_epoch_end(self, epoch, logs=None):
        noised, _  = self.val_generator[0]
        denoised = self.model(noised, training=False)
        img = self.save_image(noised, denoised, f"{epoch}")
        with self.writer.as_default():
            tf.summary.image("Denoised", [img], step=epoch)
        

    @staticmethod
    def figure_to_tf_image(fig: Figure, save_to: str):
        buf = io.BytesIO()
        plt.savefig(buf, dpi=128, format="png")
        plt.close(fig)
        val = buf.getvalue()
        with open(save_to, 'wb') as f:
            f.write(val)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        return image