from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

## A simple implementation of Multihead Attention (Previous version (GPT)) 
# - For architecture understanding purpose, not used in the final GPT implementation in this repo. 
#   See CausalSelfAttention for the optimized version.

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
        self.c_proj.GPT2_SCALE_INIT = 1
        #self.dropout = nn.Dropout(config.dropout)
        # Regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Triangular masking for auto-regressive decoding
        self.register_buffer("bias", torch.tril(torch.ones(config.context_window, config.context_window)).view(1, 1, config.context_window, config.context_window))

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
        self.c_proj.GPT2_SCALE_INIT = 1
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
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Residual connection, adds the weighted sum of attention back to the input. (Pre-LN)
        x = x + self.mlp(self.ln_2(x)) # Residual connection, adds the output of the FFW MLP back to the input. (Pre-LN)
        return x

@dataclass
class GPTConfig:
    context_window: int = 1024 # Block_size
    vocab_size: int = 50257
    n_layers: int = 12
    n_head: int = 12
    n_embed: int = 768
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

        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # Prameters initialization (same as GPT-2 paper)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT2_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5 # Scale the standard deviation of the weight initialization by the number of layers (for residual connections)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.context_window, "Cannot forward, model context window is exhausted."

        # forward the tokens and postions through the GPT model
        token_embeddings = self.transformer.wte(idx) # (B,T,n_embed)
        position_embeddings = self.transformer.wpe(torch.arange(0,T,dtype=torch.long,device=idx.device)) # (T,n_embed)
        x = token_embeddings + position_embeddings # (B,T,n_embed)
        
        #x = self.transformer.drop(x)

        # Forward the transformer blocks
        for block in self.transformer.h:
            x = block(x) # (B,T,n_embed)
            
        # Final layer norm
        x = self.transformer.ln_f(x) # (B,T,n_embed)
        logits = self.lm_head(x) # (B,T,vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # (B*T, vocab_size) and (B*T,) for cross-entropy loss
        return logits, loss

    # Loads pre-trained weights from GPT-2 model (from HuggingFace) into our GPT implementation.
    @classmethod
    def from_pretrained(cls, model_type):
        # Get the weights from HuggingFace
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading pretrained weights from HF for model: %s" % model_type)

        # Loads the hyperparameters and weights from HuggingFace GPT-2 model
        config_args = {
            'gpt2': dict(n_layers=12, n_head=12, n_embed=768), # GPT-2 small (124M parameters)
            'gpt2-medium': dict(n_layers=24, n_head=16, n_embed=1024), # GPT-2 medium (355M parameters)
            'gpt2-large': dict(n_layers=36, n_head=20, n_embed=1280), # GPT-2 large (774M parameters)
            'gpt2-xl': dict(n_layers=48, n_head=25, n_embed=1600), # GPT-2 xl (1558M parameters)
        }[model_type]
        config_args['context_window'] = 1024
        config_args['vocab_size'] = 50257

        # Initialize our GPT model with the same hyperparameters as the HuggingFace GPT-2 model
        # Get state dict from HuggingFace GPT-2 model and load it into our GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        # State dict from our GPT model
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Remove the attention bias buffer (triangular mask) since it's not a parameter

        # State dict from HuggingFace GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Load the weights from HuggingFace GPT-2 model into our GPT model
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # Remove the attention bias buffer (triangular mask) since it's not a parameter
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # Remove the attention bias buffer (triangular mask) since it's not a parameter

        # These Hugginface's GPT-2 Weights are transposed for TensorFlow, we need to transpose them back for PyTorch
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] 

        assert len(sd_keys) == len(sd_keys_hf), f"Number of parameters in our GPT model and HuggingFace GPT-2 model do not match:{len(sd_keys)} vs {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Check if the shapes are compatible for transposition
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T) # Transpose the weights back for PyTorch
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k]) # Copy the weights directly
        return model
    
# -----------------------------------------------------
# Dataloader
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # Load the text data from the input file
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        # Encoding the text using tiktoken (GPT-2's tokenizer)    
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Data loaded from input.txt, total tokens: {len(self.tokens)}")
        print(f"1 Epoch has {len(self.tokens) // (B*T)} batches of size {B} and sequence length {T}.")

        # State
        self.current_position = 0

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B, T) # Inputs
        y = buf[1:].view(B, T) # Targets (next token)
        # Updates position to the next batch
        self.current_position += B*T
        # Resets to the beginning of the data if batchs run out
        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y
# -----------------------------------------------------
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set seed for reproducibility
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed(42)

train_loader = DataLoaderLite(B=8, T=1024) # B: batch size, T: sequence length (context window)

torch.set_float32_matmul_precision('high') # Enable TF32

# Weights initialization from HuggingFace
#model = GPT.from_pretrained('gpt2')
#print("Model loaded with pretrained weights from HuggingFace GPT-2 small (124M parameters).")

# Random weights initialization
model = GPT(GPTConfig())
model.to(device) # Move the model to GPU if available, otherwise keep it on CPU
print(f"Model loaded with random initialized weights on {device}.")
#logits, loss = model(x, y)

# Optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range (50):
    t0 = time.time() 
    # Time start =======================
    # Get the next batch of data
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device) # Move the batch to the same device as the model
    # Forward pass, compute loss, backward pass, and update weights
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16): # Enable mixed precision
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # Time end =======================
    torch.cuda.synchronize() # Wait for the GPU to finish before measuring time
    t1 = time.time() 
    dt = (t1 - t0)*1000
    tps = (train_loader.B * train_loader.T) / (dt/1000)
    print(f"Batch {i}, Loss: {loss.item()}, DeltaTime: {dt:.4f} ms, Tokens/sec: {tps:.2f}")

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30
model.eval() # Set the model to evaluation mode (disables dropout)

# Test inputs
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, T)
x = tokens.to(device) # Move the input tokens to the same device as the model

# Generate
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # Run the forward pass to get the logits for the next token
    with torch.no_grad():
        logits = model(x) # (B = num_return_sequences, T, vocab_size)
        # Take the last logit
        logits = logits[:, -1, :] # (B = num_return_sequences, vocab_size)
        # Get probabilities from the logits
        probs = F.softmax(logits, dim=-1) # (B = num_return_sequences, vocab_size)
        # Top-k sampling (HF default: k=50)
        tok_probs, tok_indices = torch.topk(probs, k=50, dim=-1) # tok_probs (k, B = num_return_sequences), tok_indices (B = num_return_sequences, k)
        # Sample the next token from the top-k(50) probabilities
        next_token_idx = torch.multinomial(tok_probs, num_samples=1) # (B = num_return_sequences, 1)
        next_token = torch.gather(tok_indices, -1, next_token_idx) # (B = num_return_sequences, 1)
        # Append the predicted token to the input sequence
        x = torch.cat((x, next_token), dim=1) 

# Display the predicted outputs
for i in range(num_return_sequences):
    output_tokens = x[i, :max_length].tolist()
    output_text = enc.decode(output_tokens)
    print(f"Output {i+1}: {output_text}")
