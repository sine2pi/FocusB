
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
from torch.nn.functional import scaled_dot_product_attention

def calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True):
    # masking setup for pure decoder causal asr which typically will not require pad encoder or cross attention masking.
    batch, head, ctx, dims = q.shape
    attn_mask = None
    if mask is not None:
        mask=mask[:ctx, :ctx]
    scaled_q = q
    if temperature != 1.0 and temperature > 0:
        scaled_q = q * (1.0 / temperature)**.5
    a = scaled_dot_product_attention(scaled_q, k, v, attn_mask=attn_mask, is_causal=mask is not None and ctx > 1)     
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    return out, None

def shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

def reshape_to_output(self, attn_output, batch, ctx):
    return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims).contiguous()

def qkv_init(dims: int, head: int):
    head_dim = dims // head
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims, bias=False)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    lna = nn.LayerNorm(dims, bias=False)  
    lnb = nn.LayerNorm(head_dim, bias=False)
    return q, k, v, o, lna, lnb

def create_qkv(dims, head, q, k, v, x, xa=None):
    head_dim = dims // head
    scale = head_dim ** -0.25
    q = q(x) * scale
    k = k(x if xa is not None else xa) * scale
    v = v(x if xa is not None else xa)
    batch, ctx, _ = q.shape
    def _shape(tensor):
        return tensor.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
    return _shape(q), _shape(k), _shape(v)

def calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True):
    batch, head, ctx, dims = q.shape
    attn_mask = None
    if mask is not None:
        if mask.dim() <= 3:
            attn_mask = create_attention_mask(
                batch_size=batch,
                ctx=ctx,
                is_causal=is_causal,
                padding_mask=mask if mask.dim() > 1 else None,
                device=q.device)
        else:
            attn_mask = mask
    scaled_q = q
    if temperature != 1.0 and temperature > 0:
        scaled_q = q * (1.0 / temperature)**.5
    a = scaled_dot_product_attention(scaled_q, k, v, attn_mask=attn_mask, is_causal=is_causal if attn_mask is None else False)
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    return out, None

class LocalAttentionModule(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.query_module = nn.Linear(head_dim, head_dim)
        self.key_module = nn.Linear(head_dim, head_dim)
        self.value_module = nn.Linear(head_dim, head_dim)
        self.out_proj = nn.Linear(head_dim, head_dim)
    
    def _reshape_to_output(self, x):
        return x

class attention(nn.Module):
    def __init__(self, dims: int, head: int, max_iters: int = 3, threshold: float = 0.01, factor: float = 0.1, dropout: float = 0.1):
        super(attention, self).__init__()
        
        self.q,  self.k,  self.v,  self.o, self.lna, self.lnb = qkv_init(dims, head)
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.dropout = dropout
        self.max_iters = max_iters

        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.factor = nn.Parameter(torch.tensor(factor))
        self.attn_local = LocalAttentionModule(self.head_dim)

    def _focus(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None):

        q, k, v = create_qkv(self.dims, self.head, self.q, self.k, self.v, self.lna(x), self.lna(x if xa is None else xa))
        q=self.lnb(q)
        k=self.lnb(k)
        
        iteration = 0
        prev_attn = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        threshold = self.threshold.item()
        factor = self.factor.item()

        q_cur = q
        while iteration < self.max_iters:
            eff_span = min(q_cur.shape[2], k.shape[2], (x if xa is None else xa).shape[1])

            if eff_span == 0: 
                break

            q_iter = q_cur[:, :, :eff_span, :]
            k_iter = k[:, :, :eff_span, :]
            v_iter = v[:, :, :eff_span, :]

            q_proj = self.attn_local.query_module(q_iter)
            k_proj = self.attn_local.key_module(k_iter)
            v_proj = self.attn_local.value_module(v_iter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4: 
                    iter_mask = mask[:, :, :eff_span, :eff_span]
                elif mask.dim() == 2: 
                    iter_mask = mask[:eff_span, :eff_span]

            attn_output_iter, _ = calculate_attention(
                self.lnb(q_proj), self.lnb(k_proj), v_proj,
                mask=iter_mask,
                is_causal=True)

            out_span = self.attn_local._reshape_to_output(attn_output_iter)
            if out_span.dim() == 4:
                b, h, s, d = out_span.shape
                proj_span = self.attn_local.out_proj(out_span.view(-1, d)).view(b, h, s, -1)
            elif out_span.dim() == 3:
                b, s, d = out_span.shape
                if d == self.head_dim:
                    proj_span = self.attn_local.out_proj(out_span.view(-1, d)).view(b, 1, s, -1)
                elif d == self.head * self.head_dim:
                    proj_span = out_span.view(b, self.head, s, self.head_dim)
                else:
                    raise RuntimeError(f"Cannot reshape out_span of shape {out_span.shape} to [b, h, s, head_dim]")
            else:
                raise RuntimeError(f"Unexpected out_span shape: {out_span.shape}")

            iter_out = torch.zeros_like(q_cur)
            iter_out[:, :, :eff_span, :] = proj_span

            diff = torch.abs(iter_out - prev_attn).mean()
            dthresh = threshold + factor * diff

            if diff < dthresh and iteration > 0:
                attn_out = iter_out
                break

            prev_attn = iter_out.clone()
            q_cur = q_cur + iter_out
            attn_out = iter_out
            iteration += 1

        output = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(output), None

    def _slide_win_local(self, x: Tensor, win_size: int, span_len: int,
                         mask: Optional[Tensor] = None) -> Tensor:
        batch, ctx, dims = x.size()
        output = torch.zeros_like(x)
        num_win = (ctx + win_size - 1) // win_size

        for i in range(num_win):
            q_start = i * win_size
            q_end = min(q_start + win_size, ctx)
            q_len = q_end - q_start
            if q_len == 0: 
                continue

            kv_start = max(0, q_end - span_len)
            kv_end = q_end
            query_win = x[:, q_start:q_end, :]
            key_win = x[:, kv_start:kv_end, :]

            win_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    win_mask = mask[:, :, q_start:q_end, kv_start:kv_end]
                elif mask.dim() == 2:
                    win_mask = mask[q_start:q_end, kv_start:kv_end]

            attn_out_win, _ = self._focus(
                x=query_win,
                xa=key_win,
                mask=win_mask)

            output[:, q_start:q_end, :] = attn_out_win
        return output

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, 
                use_sliding_window: bool = False, win_size: int = 512, span_len: int = 1024) -> Tensor:
        if use_sliding_window:
            return self._slide_win_local(x, win_size, span_len, mask)
        else:
            output, _ = self._focus(x, xa, mask)
            return output

attn = attention(dims=512, head=8, max_iters=3)

x = torch.randn(2, 100, 512)
output = attn(x)

xa = torch.randn(2, 50, 512)
output = attn(x, xa=xa)

output = attn(x, use_sliding_window=True, win_size=256, span_len=512)      
   
