import os, cv2, uuid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.process import GaussianNoise

class NoiseData:

    def __init__(self, basedir: str, num_images: int) -> None:
        self.num_images = num_images
        self.basedir    = basedir

        total_images = num_images*9
        data_name = "%d" % total_images if total_images < 1000 else "%dk" % (total_images//1000)


        self.stent_dir      = os.path.join(basedir, "stents")
        self.savedir        = os.path.join(basedir, "data_%s" % data_name)
        self.annot_filename = os.path.join(basedir, "noise_annots_%s.txt" % data_name)

        os.makedirs(self.savedir, exist_ok=True)
        self.create_data()
        

    def create_data(self) -> None:
        gaussian_noise = GaussianNoise()
        annots = ""

        for entry in os.scandir(self.stent_dir):
            stent_name = entry.name
            stent = cv2.imread(entry.path, 0) / 255.
            for scale in tqdm(np.linspace(0.3, 0.5, self.num_images)):
                noised_filename   = str(uuid.uuid4()).split('-')[0]
                noised_filename   = os.path.join(self.savedir, "%s.jpg" % noised_filename)
                original_filename = os.path.join(self.stent_dir, stent_name)

                noised_stent = gaussian_noise(image=stent, scale=scale)
                noised_stent = noised_stent * 255.
                plt.imsave(noised_filename, noised_stent, cmap="gray")
                annots += "%s, %s, %s\n" % (noised_filename, original_filename, scale)
        print("%d noised images are created in %s" % (9*self.num_images, self.savedir))
        
        with open(self.annot_filename, "w") as file: file.write(annots)
