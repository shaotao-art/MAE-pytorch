import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, CenterCrop, Normalize

from einops import rearrange
from functools import partial

from cv_common_utils import read_file_lst_txt

def flower_train_t(img_size, mean, std):
    t = Compose([
        Resize(img_size),
        RandomCrop((img_size, img_size)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return t

def flower_test_t(img_size, mean, std):
    t = Compose([
        Resize(img_size),
        CenterCrop((img_size, img_size)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return t


class ImageDataset(Dataset):
    def __init__(self, root_dir, file_lst_txt=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        files = None
        if file_lst_txt is not None:
            files = read_file_lst_txt(file_lst_txt)
        
        if files is not None:
            self.image_files = [f for f in files if os.path.isfile(os.path.join(root_dir, f))]
        else:
            self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image
    

def get_shuffle_deshuffle_idx(b_s: int, 
                              seq_len: int,
                              device) -> torch.Tensor:
    """return torch tensor of shape (b_s, seq_len)
    of shuffled arange
    """
    # get shuffle and deshuffle idx
    shuffled_idx = torch.stack([torch.randperm(seq_len) for _ in range(b_s)]).to(device)
    deshuffled_idx = torch.zeros_like(shuffled_idx).long().to(device)
    for b in range(b_s):
        deshuffled_idx[b][shuffled_idx[b]] = torch.arange(seq_len).to(device)
    return shuffled_idx, deshuffled_idx

def get_label(x: torch.Tensor, 
              shuffled_idx: torch.Tensor, 
              num_mask_tokens: int,
              patch_size: int) -> torch.Tensor:
    """return torch.tensor of shape (b_s, num_mask_tokens, dim)"""
    assert len(x.shape) == 4
    b_s = x.shape[0]
    # split img of  (b_s, c, h, w) into (b_s, num_patchs, patch_size * patch_size * c)
    patches = rearrange(x, 
                        'b c (h_p p_1) (w_p p_2) -> b (h_p w_p) (p_1 p_2 c)', 
                        p_1=patch_size, 
                        p_2=patch_size)
    # shuffle patches
    shuffled_patches = torch.stack([patches[i, shuffled_idx[i], ...] for i in range(b_s)])
    return shuffled_patches[:, -num_mask_tokens:, :]

def collate_fn(batch, seq_len, num_mask_tokens, patch_size):
    imgs = torch.stack([_ for _ in batch])
    b_s = imgs.shape[0]
    shuffled_idx, deshuffled_idx = get_shuffle_deshuffle_idx(b_s, seq_len, imgs.device)
    label = get_label(imgs, shuffled_idx, num_mask_tokens, patch_size)
    return dict(img=imgs, 
                label=label, 
                shuffled_idx=shuffled_idx, 
                deshuffled_idx=deshuffled_idx)


def get_flower_train_data(data_config):
    fn = partial(collate_fn, **data_config.collate_config)
    dataset = ImageDataset(**data_config.dataset_config, transform=flower_train_t(**data_config.transform_config))
    data_loader = DataLoader(dataset=dataset, 
                             collate_fn=fn,
                            **data_config.data_loader_config, 
                            shuffle=True)
    return dataset, data_loader

def get_flower_test_data(data_config):
    fn = partial(collate_fn, **data_config.collate_config)
    dataset = ImageDataset(**data_config.dataset_config, transform=flower_test_t(**data_config.transform_config))
    data_loader = DataLoader(dataset=dataset, 
                             collate_fn=fn,
                            **data_config.data_loader_config, 
                            shuffle=False)
    return dataset, data_loader