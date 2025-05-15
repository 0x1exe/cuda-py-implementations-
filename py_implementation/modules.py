import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class MLPConfig:
    n_embed: int = 128
    bias: bool = True
    dropout: float = 0.1
    n_layer: int = 6
    n_head: int = 8
    block_size: int = 512
    vocab_size: int = 50257  

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.fc1 = nn.Linear(config.n_embed, 4*config.n_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4*config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class CausalAttention(nn.Module):  
    def __init__(self, config): 
        super().__init__()  
        
        assert config.n_embed % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head
        
        self.qkv_proj = nn.Linear(config.n_embed, 3*config.n_embed, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        
        self.flash_attn = hasattr(F, "scaled_dot_product_attention")
        if not self.flash_attn:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embed, dim=2)  
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.flash_attn:
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout.p if self.training else 0.0,  
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return self.dropout(y) if self.training else y  

class Block(nn.Module):
    def __init__(self, config , attention = CausalAttention): 
        super().__init__()  
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.attn = attention(config) 
        self.ln2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config, Block):
        super().__init__()  
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)  
        self.wpe = nn.Embedding(config.block_size, config.n_embed)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = LayerNorm(config.n_embed, config.bias)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=config.bias)
        
        self.lm_head.weight = self.wte.weight  

    def forward(self, x, targets=None):
        B, T = x.size()
        device = x.device
        
        tok_emb = self.wte(x)  # (B,T,C)
        pos = torch.arange(T, dtype=torch.long, device=device)
        pos_emb = self.wpe(pos)  # (T,C)
        
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx