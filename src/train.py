from utils import TackleWarnings
TackleWarnings()

import os, yaml, random
from argparse import ArgumentParser
from keras.optimizers import Adam

from cfg import Cfg
from callbacks import SaveDenoised

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
    
    test_generator = Cfg.get_generator(
        cfg=cfg,
        annots=annots["test"],
    )

    model = Cfg.get_model(cfg=cfg)
    runid = Cfg.get_runid(cfg=cfg)

    print("Training %s model" % model.name)

    max_epochs   = Cfg.get_max_epochs(cfg)
    lr_scheduler = Cfg.get_lr_scheduler(cfg)
    logdir       = Cfg.get_logdir(runid)
    tensorboard  = Cfg.get_tensorboard(logdir)
    
    callbacks = [
        tensorboard,
        SaveDenoised(logdir=logdir+"/validation", generator=val_generator)
    ]
    
    if lr_scheduler is not None: callbacks.append(lr_scheduler)
    
    optimizer = Cfg.get_optimizer(cfg=cfg)
    loss      = Cfg.get_loss(cfg=cfg)
    metrics   = Cfg.get_metrics(cfg=cfg)

    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    model.fit(
        train_generator,
        epochs=max_epochs,
        callbacks=callbacks,
        validation_data=val_generator
    )

    model.evaluate(test_generator)
    
    # Cfg.save_model(run_id=run_id, model=model)


    
if __name__ == "__main__":
    main()