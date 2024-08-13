"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import json
import os.path as osp

class Dictionary(ABC):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """
    dict_size : int # number of features in the dictionary
    activation_dim : int # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass
    
    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """
    def __init__(self, activation_dim, dict_size, device='cpu'):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True, device=device)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        dec_weight = t.randn_like(self.decoder.weight, device=device)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def from_pretrained(path, device='cpu'):
        """
        Load a pretrained autoencoder from a file.
        """
        dir_path, fname = osp.split(path)
        dir_path, _ = osp.split(dir_path)
        with open(osp.join(dir_path, 'config.json'), 'r') as f:
            config = json.load(f)
        state_dict = t.load(osp.join(dir_path, fname), map_location=device)
        self = AutoEncoder(config['activation_dim'], config['dictionary_size'], device=device)
        # print({k:v.shape for k,v in state_dict.items()})
        self.load_state_dict(state_dict)
        self.activation_dim = self.encoder.weight.shape[0]
        self.dict_size = self.encoder.weight.shape[1]
        return self
    
    def from_saelens(release, sae_id, device='cpu'):
        """
        Load a pretrained Saelens autoencoder from hugging face.
        """
        from sae_lens import SAE
        sae,cfg,sparsity = SAE.from_pretrained(release=release,
                                  sae_id=sae_id,
                                  device=device)
        self = AutoEncoder(cfg['d_in'], cfg['d_sae'], device=device)
        # we will assume: 'standard' for cfg.architecture
        #                 'none' for cfg.normalize_activations
        #                 'True' for cfg.apply_b_dec_to_input
        #                 'False' for cfg.finetuning_scaling_factor
        #                 'relu' for cfg.activation_fn_str

        # dictionary_learning  <-> sae_lens
        # bias                     b_dec
        # encoder.weight           W_enc
        # encoder.bias             b_enc
        # decoder.weight           W_dec
        # decoder.bias (DNE)        --

        state_dict = {'encoder.weight': sae.W_enc.data.T,
                      'encoder.bias': sae.b_enc.data,
                      'decoder.weight': sae.W_dec.data.T,
                      'bias': sae.b_dec.data}
        self.load_state_dict(state_dict)
        self.activation_dim = self.encoder.weight.shape[0]
        self.dict_size = self.encoder.weight.shape[1]
        return self

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))
    
    def decode(self, f):
        return self.decoder(f) + self.bias
    
    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None: # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat
        
        else: # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(f_ghost) # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost
            
class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """
    def __init__(self, activation_dim):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x
    
    def decode(self, f):
        return f
    
    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x