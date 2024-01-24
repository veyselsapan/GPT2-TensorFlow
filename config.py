"""
A configuration file that contains hyperparameters, 
dataset paths, and other configuration details. 
"""
from dataclasses import dataclass
@dataclass
class Config:
    d_model: int = 768
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    block_size: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
