import time, cv2, os
import numpy as np
import matplotlib.pyplot as plt



class Stents:
    filename = "dataset/CDStent.raw"
    savedir  = "dataset/stents"
    shape    = (256, 256)

    def __init__(self) -> None:
        content = self.read_raw()
        self.split(content)
        # plt.imshow(content, cmap="gray")
        # plt.show()
        
    @classmethod
    def read_raw(cls) -> np.ndarray:
        content = np.fromfile(cls.filename, dtype=np.uint16)
        max_val = int(np.sqrt(content.shape[0]))
        content = content.reshape((max_val, max_val, -1))
        return content/content.max()

    @classmethod
    def split(cls, content: np.ndarray) -> list[np.ndarray]:
        stent_width = content.shape[0]//3-10

        for row in range(3):
            for col in range(3):
                num = row*3+col+1
                stent = content[row*stent_width:(row+1)*stent_width, col*stent_width:(col+1)*stent_width]
                stent = stent*255.
                stent = cv2.resize(stent, cls.shape)
                filename = os.path.join(cls.savedir, "stent-%d.jpg" % num)
                plt.imsave(filename, stent, cmap="gray")
                print("Stent image saved at %s" % filename)