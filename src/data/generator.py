import cv2
import numpy as np
from tqdm import tqdm
from itertools import islice
from tensorflow import keras
from typing import List, Tuple

from data.process import TwoNormalize, ImageAugmentation

class ImageDataGenerator(keras.utils.Sequence):

    transformations = {
        "Augmentation": ImageAugmentation,
        "Normalize": TwoNormalize,
    }

    def __init__(
            self,
            annots_file: str,
            batch_size: int,
            input_shape: tuple,
            pipeline: dict, 
            shuffle: bool = True,
    ):
        self.batch_size   = batch_size
        self.input_shape  = tuple(input_shape)
        self.annots_file  = annots_file
        self.shuffle      = shuffle
        self.pipeline     = pipeline

        self.__get_annots()
        self.on_epoch_end()
    
    def __len__(self) -> int:
        # Number of batches per epochs
        return int(np.floor(len(self.list_annots))/self.batch_size)
    
    def on_epoch_end(self) -> None:
        # Update indexes after each epoch
        self.indexes = np.arange((len(self.list_annots)))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index: int) -> Tuple:
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_annots_temp = [self.list_annots[i] for i in indexes]

        # Generate data
        X, y = self.__data_generation(list_annots_temp=list_annots_temp)
        return X, y
    
    
    def __data_generation(self, list_annots_temp: List[str]) -> Tuple:
        # Generate data containing batch size examples
        X = np.empty((self.batch_size, *self.input_shape), dtype=np.float32)
        y = np.empty((self.batch_size, *self.input_shape), dtype=np.float32)
          
        # Generate data
        for i, (noised_filename, original_filename) in enumerate(list_annots_temp):
            noised_stent   = cv2.imread(noised_filename, 0)
            original_stent = cv2.imread(original_filename, 0)

            noised_stent   = cv2.resize(noised_stent, self.input_shape[:2])
            original_stent = cv2.resize(original_stent, self.input_shape[:2])
            
            # Data augmentation
            noised_stent, original_stent = self.__transform(images=[noised_stent, original_stent])
            
            X[i,] = noised_stent.reshape(self.input_shape)
            y[i,] = original_stent.reshape(self.input_shape)
            
        return X, y
    
    def __get_annots(self):
        self.list_annots = []
        total_lines = sum(1 for line in open(self.annots_file))
        with open(self.annots_file, 'r') as file:
            for line in tqdm(islice(file, None), total=total_lines, desc="Reading %s" % self.annots_file.split('/')[-1]):
                noised_filename, original_filename, _  = line.split(', ')
                self.list_annots.append([
                    noised_filename, original_filename
                ])
    
    def __transform(self, images: tuple[np.ndarray]) -> tuple[np.ndarray]:
        for operation in self.pipeline:
            name   = operation.get("name")
            params = operation.get("params", {})
            transformer = self.transformations.get(name)
            image = transformer(images=images, **params)
        return image
