import sys, io
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

class SaveDenoised:
    def __init__(self, logdir: str, generator) -> None:
        self.imgdir = os.path.join(logdir, "imgs")
        self.generator = generator
        self.batch_size = generator.batch_size

        os.makedirs(self.imgdir, exist_ok=True)


    def save_image(self, original_image, denoised_image, save_as: str):
        fig, axes = plt.subplots(1, 2, figsize=(8,6))
        ax1, ax2 = axes.ravel()
        ax1.imshow(original_image, cmap="gray"); ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("Original")
        ax2.imshow(denoised_image, cmap="gray"); ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("Denoised")
        fig.savefig("%s/%s.jpg"% (self.imgdir, save_as))
        plt.close()
    
    def __call__(self, bacth_num, noised, labels):
        for i in range(self.batch_size):
            self.save_image(noised[i], labels[i], "batch-%d-image-%d" % (bacth_num, i))