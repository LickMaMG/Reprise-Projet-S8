from utils import TackleWarnings
TackleWarnings()

import os, yaml, random
from argparse import ArgumentParser
from keras.optimizers import Adam

from cfg import Cfg

import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

def main() -> None:

    random.seed(Cfg.SEED)
    
    parser = ArgumentParser()
    parser.add_argument("--cfg",type=str,required=True,
        help="The config file contains the parameters that will load the database and the model for training",
    )

    args = parser.parse_args()
    cfg_filename = args.cfg

    with open(cfg_filename, "r") as file:
        cfg = yaml.safe_load(file)

    annots = Cfg.get_annots(cfg=cfg)

    train_generator = Cfg.get_generator(
        cfg=cfg,
        annots=annots["train"],
    )
    
    val_generator = Cfg.get_generator(
        cfg=cfg,
        annots=annots["val"],
    )

    model = Cfg.get_model(cfg=cfg)

    Cfg.train(
        cfg=cfg,
        model=model,
        train_gen=train_generator,
        val_gen=val_generator
    )
    
    # Cfg.save_model(run_id=run_id, model=model)


    
if __name__ == "__main__":
    main()