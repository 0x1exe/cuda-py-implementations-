import torch 
import torch.nn as nn
from einops import *
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self,d:int,base: int = 10000):
        super().__init__()
        self.base = base 
        self.d = d
        self.cos_cached = None 
        self.sin_cached = None
    
    def build_cache(self,x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        
        seq_len = x.shape[0]
        theta = 1. / (self.base ** (torch.arange(0,self.d,2).float() / self.d)).to(x.device)
        seq_idx = torch.arange(0,seq_len).float().to(x.device)


        idx_theta = torch.einsum("n,d -> nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta,idx_theta],dim=-1)

        self.cos_cached = idx_theta2.cos()[:,None,None,:]
        self.sin_cached = idx_theta2.sin()[:,None,None,:]
    
    def neg_half(self,x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:,:,:,d_2:], x[:,:,:,:d_2]],dim=-1)
    
    def forward(self,x: torch.Tensor):
        self.build_cache(x)
        neg_half_x = self.neg_half(x)
        cos_part = x * self.cos_cached[:x.shape[0]].permute(0,1,3,2)
        sin_part = x * self.sin_cached[:x.shape[0]].permute(0,1,3,2)
        rope = cos_part + sin_part * neg_half_x
        return rope


def scaled_dot_product_gqa(query,key,value,dropout = 0.0,
                           scale = None,
                           mask = None,
                           is_causal = False,
                           need_weights = False,
                           average_attn_weights = False,
                           force_grouped = False):
    
    if (mask and is_causal):
        raise ValueError('Only one of parameters: mask/is_causal should be provide, but got both')
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(f'Expected queries, keys and values to be 4-dimensional, but got shapes: Q:{query.shape},K:{key.shape},V:{value.shape}')
    
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    query = rearrange(query, "b (h g) n d -> b g h n d", g = num_head_groups)
    similarity = einsum(query,key, "b g h n d, b h s d -> b g h n s")
    if is_causal:
        mask = torch.ones((bq,nk,nk),device=query.device, dtype = torch.bool).tril_()
    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask,"b s -> b () () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () () n s")
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)
    
    attention = F.softmax(similarity, dim = -1)
    if dropout > 0.0:
        attention= F.dropout(attention, p = dropout)
    
    out = einsum(attention, value, "b g h n s, b h s d -> b g h n d")
    out = rearrange(out , "b g h n d -> b n (h g) d")

    attn_weights = None
    if need_weights:
        attn_weights = rearrange(attention, "b g h n s -> b n s (h g)")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim = 1)
    return out, attn_weights

class MultiHeadGQA(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            query_heads: int,
            kv_heads:int,
            dropout: float = 0.0,
            bias: bool = True,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
            gamma_init: float = 1.0,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads !=0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by "
                f"kv_heads ({kv_heads})"
            )
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"query_heads ({query_heads}) and kv_heads ({kv_heads})"
            )
        
        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = bias,device = device,dtype = dtype)
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = nn.Linear(embed_dim, kv_embed_dim, bias=bias, device = device,dtype = dtype)
        self.v_proj = nn.Linear(embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype)
        self.norm = None 
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, eps= layer_norm_eps,device = device, dtype = dtype)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self,query,key,value,
                need_weights: bool = False,
                is_causal: bool = False,
                average_attn_weights: bool = False,
                past_kv: tuple[torch.Tensor,torch.Tensor] = None,
                use_cache: bool = False):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)

        ##---------Apply RoPE here ----------
        q_rope=RoPE(d=q.shape[-2])
        k_rope=RoPE(d=k.shape[-2])
        q = q_rope(q)
        k = k_rope(k)
        ##-----------------------------------

        if past_kv is not None:
            past_k , past_v = past_kv
            k = torch.cat([past_k,k],dim=1)
            v= torch.cat([past_v,v],dim=1)

        
        x, attn = scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            force_grouped=False,
        )

        kv_cache = (k,v) if use_cache else None

        x = rearrange(x, "b n h d -> b n (h d)")

        if self.layer_norm:
            assert self.layer_norm != None
            x = self.norm(x)
        out = self.out_proj(x)

        return (x,attn, kv_cache) if use_cache else (x,attn)
    

