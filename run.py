from mmengine import Config

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import os
import math
import argparse

import torch
from torch import nn
import torch.nn.functional as F


from run_utils import get_callbacks, get_time_str, get_opt_lr_sch
from flower_dataset import get_flower_train_data, get_flower_test_data
from mae import TorchMae
from cv_common_utils import show_or_save_batch_img_tensor


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        ########## ================ MODEL ==================== ##############
        self.model = TorchMae(**config.model_config)
        ########## ================ MODEL ==================== ##############
                

    def training_step(self, batch, batch_idx):
        loss = self.model.train_loss(batch)
        self.log_dict({'train_loss': loss / 2.0})
        return loss
        
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if batch_idx == 0:
            imgs = batch['img']
            shuffled_idx = batch['shuffled_idx']
            b_s = shuffled_idx.shape[0]
            img_lst = []
            for b in range(b_s):
                img = imgs[b: b+1, ...]
                not_masked_idx = shuffled_idx[b][:self.config.num_patchs - self.config.num_mask_tokens]
                masked_idx = shuffled_idx[b][-self.config.num_mask_tokens:]
                pred_img = self.model.predict(img, masked_idx, not_masked_idx)
                img_lst.append(pred_img)
            vis_img_tensor = torch.cat(img_lst, dim=0)
            b_s = vis_img_tensor.shape[0]
            vis_img = show_or_save_batch_img_tensor(vis_img_tensor, int(math.sqrt(b_s)), denorm=True, mode='return')
            self.logger.experiment.add_image(tag=f'prediction-val-batch-0', 
                                img_tensor=vis_img, 
                                global_step=self.global_step,
                                dataformats='HWC',
                                )
            
                
        val_loss = self.model.train_loss(batch)
        self.log_dict({'val_loss': val_loss / 2.0})
        

    def configure_optimizers(self):
        return get_opt_lr_sch(self.config.optimizer_config, 
                              self.config.lr_sche_config,  
                              self.model)
    




def run(args):
    config = Config.fromfile(args.config)
    
    # make ckp accord to time
    time_str = get_time_str()
    config.ckp_root = '-'.join([time_str, config.ckp_root, f'[{args.run_name}]'])
    config.ckp_config['dirpath'] = config.ckp_root
    os.makedirs(config.ckp_root, exist_ok=True)
    config.run_name = args.run_name
    # logger
    
    # wandb_logger = None
    # if config.enable_wandb:
    #     wandb_logger = WandbLogger(**config.wandb_config,
    #                             name=args.wandb_run_name)
    #     wandb_logger.log_hyperparams(config)
    logger = TensorBoardLogger(save_dir=config.ckp_root,
                               name=config.run_name)
    
    # DATA
    print('getting data...')
    train_data, train_loader = get_flower_train_data(config.train_data_config)
    val_data, val_loader = get_flower_test_data(config.val_data_config)
    print(f'len train_data: {len(train_data)}, len val_loader: {len(train_loader)}.')
    print(f'len val_data: {len(val_data)}, len val_loader: {len(val_loader)}.')
    print('done.')


    # lr sche 
    if config.lr_sche_config.type in ['linear', 'cosine']:
        warm_up_epoch = config.lr_sche_config.config.warm_up_epoch
        config.lr_sche_config.config.pop('warm_up_epoch')
        config.lr_sche_config.config['num_warmup_steps'] = int(warm_up_epoch * len(train_loader))
        config.lr_sche_config.config['num_training_steps'] = config.num_ep * len(train_loader)
    
    # MODEL
    print('getting model...')
    model = Model(config)
    print(model)
    if 'load_weight_from' in config and config.load_weight_from is not None:
        # only load weights
        state_dict = torch.load(config.load_weight_from)['state_dict']
        model.load_state_dict(state_dict)
        print(f'loading weight from {config.load_weight_from}')
    print('done.')
    
    
    callbacks = get_callbacks(config.ckp_config)
    config.dump(os.path.join(config.ckp_root, 'config.py'))
    
    # TRAINING
    print('staring training...')
    resume_ckpt_path = config.resume_ckpt_path if 'resume_ckpt_path' in config else None
    trainer = pl.Trainer(accelerator=config.device,
                         max_epochs=config.num_ep,
                         callbacks=callbacks,
                         logger=logger,
                         **config.trainer_config
                         )
    
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_ckpt_path
                )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path to mmcv config file")
    parser.add_argument("--run_name", required=True, type=str, help="wandb run name")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(42)
    run(args)