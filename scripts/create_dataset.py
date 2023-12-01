import sys; sys.path.append("./src")
from argparse import ArgumentParser

from data.noise_data import NoiseData

def main():
    parser = ArgumentParser(description="Create noised images dataset")
    parser.add_argument(
        "--basedir", "-b", type=str, required=True,
        help="Directory that contains stents directory anw where will be saved noised images and annot file."

    )
    parser.add_argument(
        "--num-images", "-n", type=int, required=True,
        help="Number of images you want to create by type of stent."
    )

    args = parser.parse_args()
    basedir = args.basedir
    num_images = args.num_images

    NoiseData(basedir=basedir, num_images=num_images)

if __name__ == "__main__": main()
