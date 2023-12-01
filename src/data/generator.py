from tensorflow import keras

import numpy as np
from tensorflow import keras
from typing import List, Tuple
import cv2

from data.process import DataAugmentor

class ImageDataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            list_files,
            batch_size=2,
            # output_shape,
            input_shape: tuple = (256, 256, 1),
            shuffle: bool = True,
    ):
        self.batch_size  = batch_size
        self.input_shape = input_shape
        self.list_files  = list_files
        self.shuffle     = shuffle

        self.on_epoch_end()
    
    def __len__(self) -> int:
        # Number of batches per epochs
        return int(np.floor(len(self.list_files))/self.batch_size)
    
    def on_epoch_end(self) -> None:
        # Update indexes after each epoch
        self.indexes = np.arange((len(self.list_files)))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index: int) -> Tuple:
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_files_temp = [self.list_files[i] for i in indexes]

        # Generate data
        X, y = self.__data_generation(list_files_temp=list_files_temp)
        return X, y
    
    
    def __data_generation(self, list_files_temp: List[str]) -> Tuple:
        # Generate data containing batch size examples
        X = np.empty((self.batch_size, *self.input_shape), dtype=np.float32)
        y = np.empty((self.batch_size, *self.input_shape), dtype=np.float32)

        augmentor = DataAugmentor()
        
        
        # Generate data
        for i, (noised_filename, original_filename) in enumerate(list_files_temp):
            noised_stent = cv2.imread(noised_filename, 0)
            original_stent = cv2.imread(original_filename, 0)
            
            # Data augmentation
            noised_stent = augmentor(image=noised_stent)
            
            X[i,] = noised_stent.reshape(self.input_shape)
            y[i,] = original_stent.reshape(self.input_shape)
            
        return X, y
    
    def on_epoch_end(self) -> None:
        # Update ibdexes after each epoch
        self.indexes = np.arange((len(self.list_files)))
        if self.shuffle:
            np.random.shuffle(self.indexes)
