from utils import PathManager, TackleWarnings
PathManager("./src"); TackleWarnings()

import os, yaml, random
from argparse import ArgumentParser
from callbacks import  ConfusionMatrix
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

    model = Cfg.get_model(cfg=cfg)

    optimizer = Cfg.get_optimizer(cfg=cfg)
    loss      = Cfg.get_loss(cfg=cfg)
    metrics   = Cfg.get_metrics(cfg=cfg)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics = metrics
    )
    
    

    annots = Cfg.get_annots(cfg=cfg)


    train_generator = Cfg.get_generator(
        cfg=cfg,
        annots=annots["train"],
    )
    
    val_generator = Cfg.get_generator(
        cfg=cfg,
        annots=annots["val"],
    )
    
    max_epochs     = Cfg.get_max_epochs(cfg)
    lr_scheduler   = Cfg.get_lr_scheduler(cfg)
    run_id, logdir = Cfg.get_id_logdir(cfg)
    tensorboard    = Cfg.get_tensorboard(logdir)

    callbacks = [
        lr_scheduler,
        tensorboard
    ]

    model.fit(
        train_generator,
        epochs=max_epochs,
        callbacks=callbacks,
        validation_data = val_generator
    )
    
    Cfg.save_model(run_id=run_id, model=model)


    
if __name__ == "__main__":
    main()