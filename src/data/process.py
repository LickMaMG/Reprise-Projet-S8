import numpy as np
import albumentations as alb

class GaussianNoise:
    
    def __call__(self, image: np.ndarray, scale: float = None) -> np.ndarray:
        # scale = np.random.uniform(0.01, 0.5)
        noise = np.random.normal(loc=0, scale=scale, size=image.shape[:2])
        return noise + image

class DataAugmentor:
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        initial_shape = image.shape[:2]
        transformer = alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.CenterCrop(200, 200, p=0.5),
            alb.Rotate(limit=360, p=1),
            alb.Resize(*initial_shape, p=1)
        ])

        return transformer(image=image)["image"]
