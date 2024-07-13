device = 'cuda'

num_ep = 100
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 5e-5,
    )
)

lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # warm_up_epoch=1
    )
)



####---- model ----####
patch_size = 16
img_size = 224
mask_ratio = 0.75
num_patchs = (img_size // patch_size) ** 2
embed_dim = patch_size * patch_size * 3
num_mask_tokens = int(num_patchs * mask_ratio)

# follow vit-base
model_config = dict(
    img_size = img_size,
    patch_size = patch_size,
    mask_ratio = mask_ratio,
    torch_transformer_encoder_config = dict(
        num_layers=12,
        layer_config = dict(
            d_model=768,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            nhead=12,
            norm_first=True,
            batch_first=True,
            bias=True
        )),
    torch_transformer_decoder_config = dict(
        num_layers=2,
        layer_config = dict(
            d_model=768,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            nhead=12,
            norm_first=True,
            batch_first=True,
            bias=True
        ))
)
####---- model ----####



####---- data ----####
data_root = '/home/dmt/shao-tao-working-dir/DATA/OpenDataLab___Oxford_102_Flower/raw/jpg'
train_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    collate_config = dict(
        seq_len=num_patchs, 
        num_mask_tokens=num_mask_tokens, 
        patch_size=patch_size
    ),
    dataset_config = dict(
        root_dir=data_root,
        file_lst_txt='train.txt'
    ), 
    data_loader_config = dict(
        batch_size = 128,
        num_workers = 4,
    )
)
val_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    collate_config = dict(
        seq_len=num_patchs, 
        num_mask_tokens=num_mask_tokens, 
        patch_size=patch_size
    ),
    dataset_config = dict(
        root_dir=data_root,
        file_lst_txt='val.txt'
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 4,
    )
)
####---- data ----####


resume_ckpt_path = None
load_weight_from = None

# ckp
ckp_config = dict(
   save_last=True, 
   every_n_epochs=None,
#    monitor='val_mae',
#    mode='min',
#    filename='{epoch}-{val_mae:.3f}'
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='32',
    # val_check_interval=1, # val after k training batch 0.0-1.0, or a int
    check_val_every_n_epoch=1,
    num_sanity_val_steps=2
)


# LOGGING
enable_wandb = True
wandb_config = dict(
    project = 'backbone-exp',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'