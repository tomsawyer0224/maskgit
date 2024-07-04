import os
import torch
import torch.nn as nn
from torch import optim
import lightning as L

import random

from models import VQGAN, vector_quantizer, Transformer
from utils import save_image, plot_metrics

class VQGAN_Wrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module | dict,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        gamma: float = 0.99
    ) -> None:
        '''
        args:
            model: if is dict, it shoule be in the form (see yaml file for more details)
                {
                    'encoder_config': {'name': enc_cls_name, 'other': val,...},
                    'decoder_config': {'name': dec_cls_name, 'other': val,...},
                    'vq_config': {'name': vq_cls_name, 'other': val,...},
                    'discriminator_config': {'name': disc_cls_name, 'other': val,...},
                    'loss_config': {}
                }
        '''
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if isinstance(model, dict):
            name = model['name']
            model = {k: v for k, v in model.items() if k != 'name'}
            model = eval(name)(**model)
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
    @classmethod
    def from_pretrained(cls, checkpoint: str):
        #wrapper_model = VQGAN_Wrapper.load_from_checkpoint(checkpoint)
        wrapper_model = cls.load_from_checkpoint(checkpoint)
        return wrapper_model.model
    def configure_optimizers(self):
        vq_params = list(self.model.encoder.parameters()) + \
                    list(self.model.vq_in.parameters()) + \
                    list(self.model.vq.parameters()) + \
                    list(self.model.vq_out.parameters()) + \
                    list(self.model.decoder.parameters())
                    
        vq_optimizer = optim.AdamW(
            vq_params,
            lr = self.lr,
            weight_decay = self.weight_decay
        )
        vq_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            vq_optimizer,
            gamma = self.gamma
        )

        gan_optimizer = optim.AdamW(
            self.model.discriminator.parameters(),
            lr = self.lr,
            weight_decay = self.weight_decay
        )
        gan_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            gan_optimizer,
            gamma = self.gamma
        )
        return [vq_optimizer, gan_optimizer], [vq_lr_scheduler, gan_lr_scheduler]
    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        outputs = self.model(x)
        return outputs
    def training_step(self, batch):
        vq_optimizer, gan_optimizer = self.optimizers()
        vq_lr_scheduler, gan_lr_scheduler = self.lr_schedulers()

        global_step = self.trainer.global_step / 2
        # divide by 2 because we have 2 optimizers, each .step() method is called,
        # global_step will be increased by 1 -> real_global_step = global_step/2
        disc_factor = self.model.adopt_disc_factor(
            disc_factor = self.model.loss_config['discriminator_factor'], 
            global_step = global_step,
            threshold = self.model.loss_config['discriminator_start']
        )

        images, labels = batch

        outputs = self(images)

        losses = self.model.loss_fn(**outputs)

        vq_loss = losses['vq_loss']
        gan_loss = losses['gan_loss']

        vq_optimizer.zero_grad()
        self.manual_backward(vq_loss, retain_graph = True)
        
        gan_optimizer.zero_grad()
        self.manual_backward(gan_loss)

        vq_optimizer.step()
        gan_optimizer.step()
        
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1)%10 == 0:
            vq_lr_scheduler.step()
            gan_lr_scheduler.step()

        log = {
            f'train_{k}': v.item() for k, v in losses.items()
        }
        logit_disc_factor = {
            'train_real_logit': outputs['real_logit'].mean().item(),
            'train_fake_logit': outputs['fake_logit'].mean().item(),
            'disc_factor': disc_factor
        }
        log = log | logit_disc_factor
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
        
    def validation_step(self, batch):
        images, labels = batch

        outputs = self(images)

        losses = self.model.loss_fn(**outputs)

        vq_loss = losses['vq_loss']
        gan_loss = losses['gan_loss']

        log = {
            f'val_{k}': v for k, v in losses.items()
        }
        logit = {
            'val_real_logit': outputs['real_logit'].mean().item(),
            'val_fake_logit': outputs['fake_logit'].mean().item()
        }
        log = log | logit
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
    def test_step(self, batch):
        images, labels = batch

        outputs = self(images)

        losses = self.model.loss_fn(**outputs)

        vq_loss = losses['vq_loss']
        gan_loss = losses['gan_loss']

        log = {
            f'test_{k}': v for k, v in losses.items()
        }
        logit = {
            'test_real_logit': outputs['real_logit'].mean().item(),
            'test_fake_logit': outputs['fake_logit'].mean().item()
        }
        log = log | logit
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
    def on_validation_epoch_end(self):
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        if current_epoch % 10 == 0 or current_epoch == max_epochs-1:
            device = self.device
            log_dir = self.trainer.log_dir
            images, labels = next(iter(self.trainer.val_dataloaders))
            images = images.to(device)
            labels = labels.to(device)
            n_samples = min(len(images),2)
            org_images = images[0:n_samples]
            rec_images = self.model.reconstruct(org_images)
            
            org_images = (org_images+1)/2
            # save images
            dir_name = log_dir + '/images'
            os.makedirs(dir_name, exist_ok = True)
            file_name = f'validate_on_epoch_{current_epoch}.png'
            path = os.path.join(
                dir_name,
                file_name
            )
            save_image(
                images = [org_images.cpu(), rec_images.cpu()],
                path = path,
                titles = ['original', 'reconstructed']
            )


class Transformer_Wrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module | dict,
        vqgan_checkpoint: str,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        gamma: float = 0.99
    ) -> None:
        '''
        args:
            model: if is dict, it shoule be in the form (see yaml file for more details)
                {
                    'name': 'Transformer', # class name
                    'other param': value,
                    ...
                }
        '''
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if isinstance(model, dict):
            name = model['name']
            model = {k: v for k, v in model.items() if k != 'name'}
            model = eval(name)(**model)
        self.model = model
        self.vqgan_pretrained = VQGAN_Wrapper.from_pretrained(vqgan_checkpoint).eval()
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
    @classmethod
    def from_pretrained(cls, checkpoint: str):
        #wrapper_model = Transformer_Wrapper.load_from_checkpoint(checkpoint)
        wrapper_model = cls.load_from_checkpoint(checkpoint)
        return wrapper_model.model
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr = self.lr,
            weight_decay = self.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = self.gamma
        )
        return [optimizer], [lr_scheduler]
    def forward(
        self, 
        image: torch.Tensor, 
        label: torch.Tensor,
        masking_method: str = 'cosine',
        **kwargs
    ) -> dict[str, torch.Tensor]:
        '''
        args:
            image: (b, c, H, W)
        returns:
            logit (b, hw + 1, vocab_size)
        '''
        r = random.random() # random number in (0,1)
        mask_ratio = self.model.masking_ratio_schedule(r = r, mode = masking_method)
        with torch.no_grad():
            image_token = self.vqgan_pretrained.encode(image)[1] # (b, hw)
        image_token_mask = self.model.get_mask(
            sequence = image_token,
            ratio = mask_ratio
        )
        label_mask = self.model.get_mask(
            sequence = label,
            ratio = mask_ratio
        )
        output = self.model(
            image_token = image_token,
            image_token_mask = image_token_mask,
            label = label,
            label_mask = label_mask
        )
        return output
    def training_step(self, batch):
        optimizer = self.optimizers()
        lr_sch = self.lr_schedulers()

        image, label = batch
        output = self(image = image, label = label)
        loss = self.model.loss_fn(**output, mask_only = False)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if self.trainer.is_last_batch and (self.trainer.current_epoch+1)%10 == 0:
            lr_sch.step()
        log = {
            f'train_loss': loss.item()
        }
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
        #return loss
        
    def validation_step(self, batch):
        image, label = batch
        output = self(image = image, label = label)
        loss = self.model.loss_fn(**output, mask_only = False)
        log = {
            f'val_loss': loss.item()
        }
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
        #return loss
    def test_step(self, batch):
        image, label = batch
        output = self(image = image, label = label)
        loss = self.model.loss_fn(**output, mask_only = False)
        log = {
            f'test_loss': loss.item()
        }
        self.log_dict(
            log,
            prog_bar = True,
            on_step = True,
            on_epoch = True
        )
        #return loss
    def on_validation_epoch_end(self):
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        if current_epoch % 10 == 0 or current_epoch == max_epochs-1:
            device = self.device
            log_dir = self.trainer.log_dir
            
            n_samples = 4
            image, label = next(iter(self.trainer.val_dataloaders))
            idx = torch.randperm(len(label))[:n_samples]
            real_image = image[idx]
            real_image = (real_image+1.0)/2.0 # [-1,1] -> [0,1]
            label = label[idx]

            generated_image_token = self.model.generate_image_token(
                n_samples = n_samples,
                label = label,
                n_step = 12,
                temperature = 1.0,
                masking_method = 'cosine'
            )
            #print(f'on_validation_epoch_end')
            #print(f'generated_image_token: {generated_image_token.min()}, {generated_image_token.max()}')
            #print(f'CB_MASK_TOKEN_ID: {self.model.CB_MASK_TOKEN_ID}')
            gen_image = self.vqgan_pretrained.indices_to_image(generated_image_token)

            # save images
            dir_name = log_dir + '/images'
            os.makedirs(dir_name, exist_ok = True)
            file_name = f'validate_on_epoch_{current_epoch}.png'
            path = os.path.join(
                dir_name,
                file_name
            )
            save_image(
                images = [real_image.cpu(), gen_image.cpu()],
                path = path,
                titles = ['real image','generated']
            )

    