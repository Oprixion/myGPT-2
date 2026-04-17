from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

## A simple implementation of Multihead Attention
#class Head(nn.Module):
#    
#    def __init__(self, head_size):
#        super().__init__()
#        self.key = nn.Linear(n_embed, head_size, bias=False)
#        self.query = nn.Linear(n_embed, head_size, bias=False)
#        self.value = nn.Linear(n_embed, head_size, bias=False)
#        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#
#        self.dropout = nn.Dropout(dropout)
#
#    def forward(self, x):
#        B,T,C = x.shape
#        k = self.key(x)   # (B,T,head_size)
#        q = self.query(x) # (B,T,head_size)
#
#        # compute attention scores ("affinities")
#        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
#        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
#        wei = F.softmax(wei, dim=-1) # (B,T,T)
#        wei = self.dropout(wei)
#
#        v = self.value(x) # (B,T,head_size)
#        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
#        return out
    
#class MultiHeadAttention(nn.Module):
#
#    def __init__(self, num_heads, head_size):
#        super().__init__()
#        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#        self.proj = nn.Linear(n_embed, n_embed)
#        self.dropout = nn.Dropout(dropout)
#
#    def forward(self, x):
#        out = torch.cat([h(x) for h in self.heads], dim=-1)
#        out = self.proj(out)
#        return out
##

# PyTorch optimized implementation of Multihead Attention, including heads and batches into the tensor for better performance.
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        #self.dropout = nn.Dropout(config.dropout)
        # Regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Triangular masking for auto-regressive decoding
        self.register_buffer("bias", torch.tril(torch.ones(config.context_window, config.context_window))).view(1, 1, config.context_window, config.context_window)

    def forward(self, x):
        B, T, C = x.size() # B: batch size, T: sequence length, C: embedding dimensionality (n_embed)
        # C (n channels/embedding dimensions) = n_head * head_size
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim (Pytorch optimized implementation)
        qkv = self.c_attn(x) # (B,T,3*n_embed)
        q, k ,v = qkv.split(self.n_embed, dim=2) # (B,T,n_embed) each
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,n_head,T,head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,n_head,T,head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,n_head,T,head_size)

        # Q @ K^T attention matrix (T, T)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # (B,n_head,T,T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B,n_head,T,T)
        att = F.softmax(att, dim=-1) # (B,n_head,T,T)
        #att = self.dropout(att)

        y = att @ v # (B,n_head,T,T) @ (B,n_head,T,head_size) -> (B,n_head,T,head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side -> (B,T,C)
        # Output projection
        y = self.c_proj(y) # (B,T,C)
        #y = self.dropout(y)
        return y
    
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed) # Expand the embedding size to 4 times for the hidden layer in the MLP
        self.gelu = nn.GELU(approximate='tanh') # tanh approximation is used to replicate GPT-2. Just GELU(approximation = 'none') is also fine.
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed) # Project back to the original embedding size
        #self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        #x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.mha = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x)) # Residual connection, adds the weighted sum of attention back to the input. (Pre-LN)
        x = x + self.mlp(self.ln2(x)) # Residual connection, adds the output of the FFW MLP back to the input. (Pre-LN)
        return x

@dataclass
class GPTConfig:
    context_window: int = 1024 # Block_size
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_embed: int = 384
    dropout: float = 0.2

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # Token Embeddings
            wpe = nn.Embedding(config.context_window, config.n_embed), # Position Embeddings
            #drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), # List of transformer blocks (MHA + FFW)
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # Language modeling head (final output layer)

