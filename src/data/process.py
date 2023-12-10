import numpy as np
import albumentations as alb

class GaussianNoise:
    
    def __call__(self, image: np.ndarray, scale: float = None) -> np.ndarray:
        # scale = np.random.uniform(0.01, 0.5)
        noise = np.random.normal(loc=0, scale=scale, size=image.shape[:2])
        return noise + image

class DataAugmentor:
    
    def __call__(self, images: np.ndarray) -> np.ndarray:
        initial_shape = images[0].shape[:2]
        width = initial_shape[0]
        kwargs = {"image": images[0], "image1": images[1]}
        transformer = alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.CenterCrop(width-width//10, width-width//10, p=0.5),
            alb.Rotate(limit=360, border_mode=1),
            alb.Resize(*initial_shape, p=1)
        ], additional_targets={"image":"image", "image1":"image"}, p=1)

        transformed = transformer(**kwargs)

        return list(map(lambda name: transformed[name], kwargs))
