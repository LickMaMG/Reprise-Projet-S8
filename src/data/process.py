import numpy as np
import albumentations as alb

class Normalize:

    def __new__(cls, image: np.ndarray, **kwargs)-> np.ndarray:
        return image.astype(np.float32)/255.

class TwoNormalize:

    def __new__(cls, images: list[np.ndarray]) -> list[np.ndarray]:
        return [Normalize(img) for img in images]

class GaussianNoise:
    def __new__(cls, image: np.ndarray, **kwargs)-> np.ndarray:
        scale = kwargs.get("scale")
        noise = np.random.normal(loc=0, scale=scale, size=image.shape[:2])
        return noise + image


class ImageAugmentation:
    p = 0.1
    def __new__(cls, images: list[np.ndarray], **kwargs) -> list[np.ndarray]:
        return cls.__transform(images=images, **kwargs)
    
    @classmethod
    def __transform(cls, images, **kwargs):
        # Return an augmentation
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