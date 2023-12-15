import os, cv2, uuid
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data.process import GaussianNoise

class NoiseData:

    def __init__(self, basedir: str, num_images: int) -> None:
        self.num_images = num_images
        self.basedir    = basedir

        total_images = num_images*9
        data_name = "%d" % total_images if total_images < 1000 else "%dk" % (total_images//1000)


        self.stent_dir      = os.path.join(basedir, "stents")
        self.savedir        = os.path.join(basedir, "data-%s" % data_name)
        self.annot_filename = os.path.join(basedir, "noise-annots-%s" % data_name)

        os.makedirs(self.savedir, exist_ok=True)
        self.create_data()
        

    def create_data(self) -> None:
        annots = []

        for entry in os.scandir(self.stent_dir):
            stent_name = entry.name
            stent = cv2.imread(entry.path, 0) / 255.
            for scale in tqdm(np.linspace(0.3, 0.5, self.num_images)):
                noised_filename   = str(uuid.uuid4()).split('-')[0]
                noised_filename   = os.path.join(self.savedir, "%s.jpg" % noised_filename)
                original_filename = os.path.join(self.stent_dir, stent_name)

                noised_stent = GaussianNoise(image=stent, **{"scale": scale})
                noised_stent = noised_stent * 255.
                plt.imsave(noised_filename, noised_stent, cmap="gray")
                annots.append("%s, %s, %s\n" % (noised_filename, original_filename, scale))
        print("\n%d noised images are created in %s" % (9*self.num_images, self.savedir))
        train_annots, val_annots = train_test_split(annots, test_size=0.2, random_state=42)
        val_annots, test_annots = train_test_split(val_annots, test_size=0.5, random_state=42)
        
        for name, dataset in [("train", train_annots), ("validation", val_annots), ("test", test_annots)]:
            print("%s : %d" % (name.ljust(10), len(dataset)))
            dataset = ''.join(dataset)
            with open(self.annot_filename+"-%s.txt" % name, 'w') as file:
                file.write(dataset)
