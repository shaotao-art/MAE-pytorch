import torch
from torch import nn
from einops import rearrange

from typing import Dict


class VitStem(nn.Module):
    def __init__(self, 
                 seq_len: int,
                 embed_dim: int,
                 patch_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dim = embed_dim
        
        self.positional_encoding = nn.Embedding(num_embeddings=seq_len, 
                                                embedding_dim=self.dim)
        self.patch_embeding = nn.Conv2d(3, 
                                        self.dim,
                                        kernel_size=patch_size, 
                                        stride=patch_size)

    
    def forward(self, x):
        assert len(x.shape) == 4
        # expect x of shape (b, 3, h, w)
        x = self.patch_embeding(x) # (b, dim, patch_h, patch_w)
        x = rearrange(x, 
                    'b d p_h p_w -> b (p_h p_w) d') # (b, num_patches, patch_dim)
        # add positional encoding
        patches = x + self.positional_encoding(torch.arange(self.seq_len).to(x.device))
        return patches



class TorchMae(nn.Module):
    def __init__(self, 
                 torch_transformer_encoder_config: Dict, 
                 torch_transformer_decoder_config: Dict,
                 img_size: int,
                 patch_size: int,
                 mask_ratio: float) -> None:
        super().__init__()
        seq_len = (img_size // patch_size) ** 2
        num_mask_tokens = int(seq_len * mask_ratio)
        dim = torch_transformer_encoder_config['layer_config']['d_model']
        dec_dim = patch_size * patch_size * 3
        
        self.patch_size = patch_size
        self.num_mask_tokens = num_mask_tokens
        self.seq_len = seq_len
        
        self.mask_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        
        self.mae_stem = VitStem(seq_len=seq_len, 
                                embed_dim=dim,
                                patch_size=patch_size, 
                                )
        
        # MAE encoder
        encoder_layer = nn.TransformerEncoderLayer(**torch_transformer_encoder_config['layer_config'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         torch_transformer_encoder_config['num_layers'])
        
        # MAE decoder
        decoder_layer = nn.TransformerEncoderLayer(**torch_transformer_decoder_config['layer_config'])
        self.decoder = nn.TransformerEncoder(decoder_layer, 
                                            torch_transformer_decoder_config['num_layers'])
        self.dec_head = nn.Linear(dim, dec_dim)
        
        
    def train_loss(self, batch):
        x = batch['img']
        shuffled_idx = batch['shuffled_idx']
        label = batch['label']
        
        # expect x shape (b, 3, h, w)
        assert len(x.shape) == 4
        b_s = x.shape[0]
        
        patches = self.mae_stem(x) # (b, num_patches, dim)
        
        # mae encoder
        # shuffle patches and cut 
        patches = torch.stack([patches[i, shuffled_idx[i], ...] for i in range(b_s)])
        patches = patches[:, :self.seq_len - self.num_mask_tokens, :] # (b, num_patches_lefted, dim)
        enc_out = self.transformer_encoder(patches)
        
        
        # mae decoder
        # add mask tokens (b, num_patches_left+num_mask_tokens, dim) -> (b, num_patches, dim)
        patches = torch.cat([enc_out, self.mask_token.repeat(b_s, self.num_mask_tokens, 1)], dim=1)
        
        # add positional encoding
        patches = patches + self.mae_stem.positional_encoding(shuffled_idx)
        dec_out = self.dec_head(self.decoder(patches))
        
        loss = torch.nn.functional.l1_loss(dec_out[:, -self.num_mask_tokens:, :], label)
        return loss
        
    @torch.no_grad()
    def predict(self, x, masked_idx, not_masked_idx):
        assert x.shape[0] == 1
        # mae encoder
        b, c, h, w = x.shape
        patches = self.mae_stem(x) # (b, num_patches, dim)

        # get only patches not masked
        patches = patches[:, not_masked_idx, :] # (b, num_not_masked_tokens, dim)
        enc_out = self.transformer_encoder(patches)
        
        
        # mae decoder
        # add mask tokens
        masked_num = len(masked_idx)
        patches = torch.cat([enc_out, self.mask_token.repeat(b, masked_num, 1)], dim=1) # (b, num_patches, dim)
        
        # add positional encoding
        patches = patches + self.mae_stem.positional_encoding(torch.cat([not_masked_idx, masked_idx]).unsqueeze(0))
        dec_out = self.dec_head(self.decoder(patches))[:, -masked_num:, :]
        
        
        # decode result into 2d img, 
        # patches not masked use origin patch intead of prediction
        dec_img = rearrange(x, 
                    'b c (h_p p_1) (w_p p_2) -> b (h_p w_p) (p_1 p_2 c)', 
                    p_1=self.patch_size, 
                    p_2=self.patch_size)
        dec_img[:, masked_idx, :] = dec_out
        
        inp_img = dec_img.clone()
        inp_img[:, masked_idx, :] = 0
        inp_img = rearrange(inp_img, 
                    'b (h_p w_p) (p_1 p_2 c) -> b c (h_p p_1) (w_p p_2)', 
                    p_1=self.patch_size, 
                    p_2=self.patch_size,
                    h_p=h//self.patch_size,
                    w_p=w//self.patch_size)
        
        dec_img = rearrange(dec_img, 
                         'b (h_p w_p) (p_1 p_2 c) -> b c (h_p p_1) (w_p p_2)', 
                         p_1=self.patch_size, 
                         p_2=self.patch_size,
                         h_p=h//self.patch_size,
                         w_p=w//self.patch_size)
        return dict(inp_img=inp_img, 
                    dec_img=dec_img)
        