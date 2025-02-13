import torch
import yaml
import argparse
import os
import lightning as L

import datasets
from utils import save_image, plot_metrics


from models import VQGAN_Wrapper, VQGAN, Transformer_Wrapper, Transformer


class TrainingVQGAN:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            configs = yaml.safe_load(file)
        self.configs = configs

    def run(
        self, max_epochs: int = None, ckpt_path: str = None, limit_batches: bool = False
    ):
        VQGAN_config = self.configs["VQGAN_config"]
        dataset_config = self.configs["dataset_config"]
        training_config = self.configs["training_config"]
        if max_epochs:
            training_config["trainer_config"]["max_epochs"] = max_epochs
        if ckpt_path:
            training_config["fit_config"]["ckpt_path"] = ckpt_path

        # VQGAN model
        vqgan = VQGAN_config
        # vqgan = VQGAN(**VQGAN_config)

        # dataset
        datamodule = datasets.LitImageGenerationDataset(**dataset_config)

        # Wrapper model
        optimizer_config = training_config["optimizer_config"]
        wrapper_model = VQGAN_Wrapper(model=vqgan, **optimizer_config)

        # trainer config
        trainer_config = training_config["trainer_config"]
        # ---logger
        if trainer_config["logger"]:
            trainer_config["logger"] = L.pytorch.loggers.CSVLogger(
                save_dir=trainer_config["default_root_dir"], name="logs"
            )

        # for reproducibility: seed_everything, deterministic
        L.seed_everything(42, workers=True)
        if limit_batches:  # for testing purpose
            limit_batches = {
                "limit_train_batches": 2,
                "limit_test_batches": 2,
                "limit_val_batches": 2,
            }
        else:
            limit_batches = {}
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
        trainer = L.Trainer(
            **trainer_config,
            deterministic=True,
            enable_checkpointing=False,
            callbacks=[lr_monitor],
            **limit_batches,
        )

        # training phase
        fit_config = training_config["fit_config"]
        trainer.fit(model=wrapper_model, train_dataloaders=datamodule, **fit_config)
        # ---plot metric curves
        if trainer_config["logger"]:
            plot_metrics(trainer.log_dir)

        # testing phase
        trainer.test(model=wrapper_model, dataloaders=datamodule)

        # save the last checkpoint manually if set enable_checkpointing = False
        default_root_dir = trainer_config["default_root_dir"]
        checkpoint_dir = os.path.join(default_root_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        epoch = trainer.current_epoch - 1
        step = trainer.global_step // 2
        checkpoint_name = f"epoch={epoch}-step={step}.ckpt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        trainer.save_checkpoint(checkpoint_path)

        self.test_on_testdataset(vqgan=wrapper_model.model, trainer=trainer)

    def test_on_testdataset(self, vqgan, trainer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_dir = trainer.log_dir

        images, labels = next(iter(trainer.test_dataloaders))
        images = images.to(device)
        labels = labels.to(device)
        vqgan.to(device)

        n_samples = min(len(images), 2)
        org_images = images[0:n_samples]
        rec_images = vqgan.reconstruct(org_images)

        org_images = (org_images + 1) / 2
        # save images
        dir_name = log_dir + "/images"
        os.makedirs(dir_name, exist_ok=True)
        file_name = f"test_on_epoch_{trainer.max_epochs-1}.png"
        path = os.path.join(dir_name, file_name)
        save_image(
            images=[org_images.cpu(), rec_images.cpu()],
            path=path,
            titles=["original", "reconstructed"],
        )


class TrainingTransformer:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            configs = yaml.safe_load(file)
        self.configs = configs

    def run(
        self, max_epochs: int = None, ckpt_path: str = None, limit_batches: bool = False
    ):
        vqgan_checkpoint = self.configs["vqgan_checkpoint"]
        Transformer_config = self.configs["Transformer_config"]
        dataset_config = self.configs["dataset_config"]
        training_config = self.configs["training_config"]
        if max_epochs:
            training_config["trainer_config"]["max_epochs"] = max_epochs
        if ckpt_path:
            training_config["fit_config"]["ckpt_path"] = ckpt_path

        # Transformer model
        transfomer_model = Transformer_config

        # dataset
        datamodule = datasets.LitImageGenerationDataset(**dataset_config)

        # Wrapper model
        optimizer_config = training_config["optimizer_config"]
        wrapper_model = Transformer_Wrapper(
            model=transfomer_model,
            vqgan_checkpoint=vqgan_checkpoint,
            **optimizer_config,
        )

        # trainer config
        trainer_config = training_config["trainer_config"]
        # ---logger
        if trainer_config["logger"]:
            trainer_config["logger"] = L.pytorch.loggers.CSVLogger(
                save_dir=trainer_config["default_root_dir"], name="logs"
            )

        # for reproducibility: seed_everything, deterministic
        L.seed_everything(42, workers=True)
        if limit_batches:  # for testing purpose
            limit_batches = {
                "limit_train_batches": 2,
                "limit_test_batches": 2,
                "limit_val_batches": 2,
            }
        else:
            limit_batches = {}
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
        trainer = L.Trainer(
            **trainer_config,
            deterministic=True,
            enable_checkpointing=False,
            callbacks=[lr_monitor],
            **limit_batches,
        )

        # training phase
        fit_config = training_config["fit_config"]
        trainer.fit(model=wrapper_model, train_dataloaders=datamodule, **fit_config)
        # ---plot metric curves
        if trainer_config["logger"]:
            plot_metrics(trainer.log_dir)

        # testing phase
        trainer.test(model=wrapper_model, dataloaders=datamodule)

        # save the last checkpoint manually if set enable_checkpointing = False
        default_root_dir = trainer_config["default_root_dir"]
        checkpoint_dir = os.path.join(default_root_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = (
            f"epoch={trainer.current_epoch-1}-step={trainer.global_step}.ckpt"
        )
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        trainer.save_checkpoint(checkpoint_path)

        self.test_on_testdataset(transformer_wrapper=wrapper_model, trainer=trainer)

    def test_on_testdataset(self, transformer_wrapper, trainer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_dir = trainer.log_dir

        transformer_wrapper.to(device)
        image, label = next(iter(trainer.test_dataloaders))

        n_samples = 4
        idx = torch.randperm(len(label))[:n_samples]
        real_image = image[idx]
        real_image = (real_image + 1.0) / 2.0  # [-1,1] -> [0,1]
        label = label[idx]
        generated_image_token = transformer_wrapper.model.generate_image_token(
            n_samples=n_samples,
            label=label,
            n_step=12,
            temperature=1.0,
            masking_method="cosine",
        )
        gen_image = transformer_wrapper.vqgan_pretrained.indices_to_image(
            generated_image_token
        )
        # save images
        dir_name = log_dir + "/images"
        os.makedirs(dir_name, exist_ok=True)
        file_name = f"test_on_epoch_{trainer.max_epochs-1}.png"
        path = os.path.join(dir_name, file_name)
        save_image(
            images=[real_image.cpu(), gen_image.cpu()],
            path=path,
            titles=["real image", "generated"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--limit_batches", type=bool, default=False)

    args = parser.parse_args()
    assert args.phase in ["vqgan", "transformer"]
    if args.phase == "vqgan":
        training = TrainingVQGAN(args.config_file)
    else:
        training = TrainingTransformer(args.config_file)
    training.run(
        max_epochs=args.max_epochs,
        ckpt_path=args.ckpt_path,
        limit_batches=args.limit_batches,
    )
