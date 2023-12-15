from utils import TackleWarnings
TackleWarnings()

import time, os
from tensorflow import keras
from sklearn.model_selection import train_test_split

from model.unet import Unet
# from model.pca import CustomPCA
from metrics import PeakSignalNoiseRatio
from data.generator import ImageDataGenerator
from callbacks import ExpSchedule, WarmupCosineSchedule, CosineSchedule


class Cfg:
    MODELS_DIR = "models"
    SEED = 42

    DATASETS = {
        "generator": ImageDataGenerator,
    }

    MODELS = {
        "unet": Unet,
        # "pca": CustomPCA,
    }

    LOSSES = {
        "categorical_crossentropy":        "categorical_crossentropy",
        "sparse_categorical_crossentropy": "sparse_categorical_crossentropy",
        "mse": "mse",
        "mae": "mae",
    }


    OPTIMIZERS = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
    }

    METRICS = {
        "mae":"mae",
        "mse":"mse",
        "psnr": PeakSignalNoiseRatio
    }

    SCHEDULERS = {
        "ExpSchedule":          ExpSchedule,
        "WarmupCosineSchedule": WarmupCosineSchedule,
        "CosineSchedule":       CosineSchedule,
    }

    @classmethod
    def get_data(cls, cfg: dict):
        name   = cfg.get("dataset").get("name")
        params = cfg.get("dataset").get("params")
        return cls.DATASETS.get(name)(**params)()
    
    @classmethod
    def get_generator(cls, cfg: dict, set_type: str):
        data_args = cfg.get("dataset")
        name   = data_args.get("name")
        params = data_args.get("params")
        params["annots_file"] = data_args.get("annots_files").get(set_type)
        if "pipelines" in data_args:
            params["pipeline"] = data_args.get("pipelines").get(set_type)
        return  cls.DATASETS.get(name)(**params)

    @classmethod
    def get_model(cls, cfg: dict):
        name   = cfg.get("model").get("name")
        params = cfg.get("model").get("params")
        model  = cls.MODELS.get(name)(**params)
        return model

    @classmethod
    def get_optimizer(cls, cfg: dict):
        opt_args = cfg.get("training").get("optimizer")
        name       = opt_args.get("name")
        params     = opt_args.get("params", None)
        optimizer  = cls.OPTIMIZERS.get(name)
        if params is not None: optimizer = optimizer(**params)

        return optimizer
    
    
    @classmethod
    def get_loss(cls, cfg: dict):
        loss_args = cfg.get("training").get("loss")
        losses = []
        for item in loss_args:
            params = item.get("params", None)
            loss = cls.LOSSES.get(item.get("name"))
            if params is not None: loss = loss(**params)
            losses.append(loss)
        return losses
    
    @classmethod
    def get_metrics(cls, cfg: dict):
        metric_args = cfg.get("training").get("metrics")
        metrics = []
        if metric_args is not None:
            for item in metric_args:
                params = item.get("params", None)
                metric = cls.METRICS.get(item.get("name"))
                if params is not None: metric = metric(**params)
                metrics.append(metric)
        return metrics

    @classmethod
    def get_custom_callbacks(cls, cfg: dict, logdir, val_generator):
        callbacks = []
        custom_callbacks = cfg.get("training").get("custom_callbacks")
        if custom_callbacks is not None:
            for item in custom_callbacks:
                name = item.get("name")
                callback = cls.CALLBACKS.get(name)
                callbacks.append(
                    callback(logdir=logdir, val_generator=val_generator)
                )
        return callbacks

    @staticmethod
    def get_max_epochs(cfg: dict):
        train_args = cfg.get("training")
        max_epochs = train_args.get("epochs")
        return max_epochs
    
    @classmethod
    def get_lr_scheduler(cls, cfg: dict):
        train_args   = cfg.get("training")
        name         = train_args.get("lr_scheduler").get("name")
        params       = train_args.get("lr_scheduler").get("params")
        lr_scheluder = cls.SCHEDULERS.get(name)(**params)
        return lr_scheluder

    @staticmethod
    def get_tensorboard(logdir: str):
        return keras.callbacks.TensorBoard(log_dir=logdir)

    @staticmethod
    def get_id_logdir(cfg: dict):
        train_name = cfg.get("name")
        run_id     = time.strftime("{}_%Y_%m_%d-%H_%M_%S".format(train_name))
        logdir     = os.path.join("./logs/", run_id)
        return run_id, logdir

    @staticmethod
    def get_runid(cfg: dict):
        training_name = cfg.get("name")
        runid         = time.strftime("{}_%Y_%m_%d-%H_%M_%S".format(training_name))
        return runid
    
    @staticmethod
    def get_logdir(runid: str):
        return os.path.join("./logs/", runid)

    @classmethod
    def save_model(cls, run_id, model):
        os.makedirs(Cfg.MODELS_DIR, exist_ok=True)
        model_folder = os.path.join(Cfg.MODELS_DIR, run_id)
        os.makedirs(model_folder, exist_ok=True)
        model_filename = os.path.join(model_folder, run_id)
        model.save_weights(model_filename)


    
