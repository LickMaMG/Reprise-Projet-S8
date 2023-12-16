from utils import TackleWarnings
TackleWarnings()

import yaml, random, argparse
from argparse import ArgumentParser

from cfg import Cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with hybrid configuration.")
    parser.add_argument("--cfg",type=str,required=True,
        help="The config file contains the parameters that will load the database and the model for training",
    )
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--data", type=str, help="Data num. Look at datasets pattern. Ex for data-1k, you should use 1k")
    parser.add_argument("--learning_rate", type=float, help="learning rate")

    return parser.parse_args()

def main() -> None:

    random.seed(Cfg.SEED)
    
    args = parse_args()
    
    cfg_filename = args.cfg

    with open(cfg_filename, "r") as file:
        cfg = yaml.safe_load(file)

    if args.batch_size:
        cfg["dataset"]["params"]["batch_size"] = args.batch_size
    if args.data:
        cfg["dataset"]["annots_files"] = {
            set_name: "dataset/noise-annots-%s-%s.txt" % (args.data, set_name)
            for set_name in cfg["dataset"]["annots_files"]
        }
    if args.learning_rate:
        cfg["training"]["optimizer"]["params"]["learning_rate"] = args.learning_rate
    
    bs = cfg["dataset"]["params"]["batch_size"]
    lr = cfg["training"]["optimizer"]["params"]["learning_rate"]
    data_name = cfg["dataset"]["annots_files"]["train"].split("annots-")[-1].split("-train.txt")[0]
    cfg["name"] = "unet--noise-images-%s--bs%d-lr%s" % (data_name, bs, str(lr))

    print("\nCreating pipelines, generators and building model...")

    train_generator = Cfg.get_generator(
        set_type="train",
        cfg=cfg,
    )

    val_generator = Cfg.get_generator(
        set_type="validation",
        cfg=cfg,
    )

    test_generator = Cfg.get_generator(
        set_type="test",
        cfg=cfg,
    )

    optimizer = Cfg.get_optimizer(cfg)
    loss      = Cfg.get_loss(cfg)
    metrics   = Cfg.get_metrics(cfg)
    
    
    max_epochs     = Cfg.get_max_epochs(cfg)
    lr_scheduler   = Cfg.get_lr_scheduler(cfg)
    run_id, logdir = Cfg.get_id_logdir(cfg)
    tensorboard    = Cfg.get_tensorboard(logdir)

    callbacks = [tensorboard]
    callbacks.extend(lr_scheduler)
    callbacks.extend(
        Cfg.get_custom_callbacks(
            cfg=cfg,
            logdir=f"{logdir}/validation",
            val_generator=val_generator,
        )
    )    
    
    model = Cfg.get_model(cfg)
    print("Training %s ..." % cfg.get("name"))
    
    model.compile(
        optimizer = optimizer,
        loss      = loss,
        metrics   = metrics
    )
    
    model.fit(
        train_generator,
        epochs=max_epochs,
        callbacks=callbacks,
        validation_data=val_generator
    )

    evaluations = model.evaluate(test_generator)
    for i, metric in enumerate(loss + metrics):
        if not isinstance(metric, str): metric = metric.name
        print("%s : %.3f" % (metric.ljust(5), evaluations[i]))
    # Cfg.save_model(run_id=run_id, model=model)


    
if __name__ == "__main__":
    main()