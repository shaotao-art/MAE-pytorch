import torch
from torch import nn
from einops import rearrange


class VitStem(nn.Module):
    def __init__(self, 
                 seq_len: int,
                 patch_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dim = patch_size * patch_size * 3
        self.patch_size = patch_size
        
        self.positional_encoding = nn.Embedding(num_embeddings=seq_len, 
                                                embedding_dim=self.dim)
        self.patch_embeding = nn.Conv2d(3, 
                                        self.dim,
                                        kernel_size=patch_size, 
                                        stride=patch_size)
    
    def forward(self, x):
        # expect x (b, 3, h, w)
        assert len(x.shape) == 4
        x = self.patch_embeding(x) # (b, dim, patch_h, patch_w)
        x = rearrange(x, 
                    'b d p_h p_w -> b (p_h p_w) d') # (b, num_patches, patch_dim)
        x = x + self.positional_encoding(torch.arange(self.seq_len))
        return x

class TorchVit(nn.Module):
    def __init__(self, 
                 torch_transformer_encoder_config, 
                 img_size: int,
                 patch_size: int) -> None:
        super().__init__()
        seq_len = (img_size // patch_size) ** 2
        
        self.vit_stem = VitStem(seq_len=seq_len, patch_size=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(**torch_transformer_encoder_config['layer_config'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         torch_transformer_encoder_config['num_layers'])
        
    
    def forward(self, x):
        assert len(x.shape) == 4
        x = self.vit_stem(x)
        x = self.transformer_encoder(x)
        return x