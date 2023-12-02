from utils import TackleWarnings
TackleWarnings()

import time, os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from callbacks import ExpSchedule, WarmupCosineSchedule, CosineSchedule
# from model.unet import Unet
from model.pca import CustomPCA
from data.generator import ImageDataGenerator


class Cfg:
    MODELS_DIR = "models"
    SEED = 42

    DATASETS = {
        "generator": ImageDataGenerator,
    }

    MODELS = {
        "unet": "Unet", #! impelment
        "pca": CustomPCA,
    }

    LOSSES = {
        "categorical_crossentropy":        "categorical_crossentropy",
        "sparse_categorical_crossentropy": "sparse_categorical_crossentropy",
        "mse": "mse"
    }


    OPTIMIZERS = {
        "adam": "adam",
    }

    METRICS = {
        "mae":"mae",
        "psnr":"PeakSignalNoiseRatio" #! implement
    }

    SCHEDULERS = {
        "ExpSchedule":          ExpSchedule,
        "WarmupCosineSchedule": WarmupCosineSchedule,
        "CosineSchedule":       CosineSchedule,
    }


    @classmethod
    def get_generator(cls, cfg: dict, annots: list):
        name   = cfg.get("dataset").get("name")
        params = cfg.get("dataset").get("params")
        return  cls.DATASETS.get(name)(
            list_files = annots,
            **params
        )
    
    @staticmethod
    def get_annots(cfg: dict):
        annot_filename = cfg.get("dataset").get("annot_filename")
        with open(annot_filename) as file:
            annots = file.read().split("\n")
        annots = [line.split(', ')[:2] for line in annots if len(line)>0]
        
        train_annots, test_annots = train_test_split(annots, test_size=0.2, random_state=Cfg.SEED)
        val_annots, test_annots   = train_test_split(test_annots, test_size=0.5, random_state=Cfg.SEED)

        annots = {
            "train": train_annots,
            "val":   val_annots,
            "test":  test_annots,
        }

        return annots

    @classmethod
    def get_model(cls, cfg: dict):
        model_name    = cfg.get("model").get("name")
        model_params  = cfg.get("model").get("params")
        model         = cls.MODELS.get(model_name)(**model_params)
        return model

    @classmethod
    def get_optimizer(cls, cfg: dict):
        train_args = cfg.get("training")
        name       = train_args.get("optimizer").get("name")
        params     = train_args.get("optimizer").get("params", None)
        optimizer  = cls.OPTIMIZERS.get(name)
        if params: optimizer = optimizer(**params)

        return optimizer
    
    @classmethod
    def get_loss(cls, cfg: dict):
        train_args = cfg.get("training")
        name       = train_args.get("loss").get("name")
        params     = train_args.get("loss").get("params", None)
        loss       = cls.LOSSES.get(name)
        if params: loss = loss(**params)
        return  loss
    

    @classmethod
    def get_metrics(cls, cfg: dict):
        train_args = cfg.get("training")
        metrics = []
        for item in train_args.get("metrics"):
            metric_params = item.get("params", None)
            metric = cls.METRICS.get(item.get("name"))
            if metric_params: metric = metric(**metric_params)
            metrics.append(metric)
        return metrics

    @staticmethod
    def get_max_epochs(cfg: dict):
        training_params = cfg.get("training")
        max_epochs      = training_params.get("epochs")
        return max_epochs
    
    @classmethod
    def get_lr_scheduler(cls, cfg: dict):
        training_params     = cfg.get("training")
        lr_scheluder_name   = training_params.get("lr_scheduler").get("name")
        lr_scheluder_params = training_params.get("lr_scheduler").get("params")
        lr_scheluder        = cls.SCHEDULERS.get(lr_scheluder_name)(**lr_scheluder_params)
        return lr_scheluder

    @staticmethod
    def get_tensorboard(logdir: str):
        return keras.callbacks.TensorBoard(log_dir=logdir)

    @staticmethod
    def get_runid(cfg: dict):
        training_name = cfg.get("name")
        runid         = time.strftime("{}_%Y_%m_%d-%H_%M_%S".format(training_name))
        return runid
    
    @staticmethod
    def get_logdir(runid: str):
        return os.path.join("./logs/", runid)
        
    @staticmethod
    def train(cfg: dict, model, train_gen, val_gen, runid):
        print("Training %s model" % model.name)

        if isinstance(model, keras.Model):
            max_epochs   = Cfg.get_max_epochs(cfg)
            lr_scheduler = Cfg.get_lr_scheduler(cfg)
            logdir       = Cfg.get_logdir(runid)
            tensorboard  = Cfg.get_tensorboard(logdir)

            callbacks = [
                lr_scheduler,
                tensorboard
            ]

            optimizer = Cfg.get_optimizer(cfg=cfg)
            loss      = Cfg.get_loss(cfg=cfg)
            metrics   = Cfg.get_metrics(cfg=cfg)

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics = metrics
            )

            model.fit(
                train_gen,
                epochs=max_epochs,
                callbacks=callbacks,
                validation_data=val_gen
            )
        else:
            if isinstance(model, CustomPCA):
                model.fit(train_gen=train_gen)

    @staticmethod
    def evaluate(model, test_gen, runid):
        print("\nEvaluate %s model" % model.name)
        if isinstance(model, keras.Model):
            model.evaluate(test_gen)
        else:
            if isinstance(model, CustomPCA):
                model.evaluate(logdir="denoised_images/%s" % runid, generator=test_gen)
    

    @classmethod
    def save_model(cls, run_id, model):
        os.makedirs(Cfg.MODELS_DIR, exist_ok=True)
        model_folder = os.path.join(Cfg.MODELS_DIR, run_id)
        os.makedirs(model_folder, exist_ok=True)
        model_filename = os.path.join(model_folder, run_id)
        model.save_weights(model_filename)


    
