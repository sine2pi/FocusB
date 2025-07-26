
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
from echoutils import *

def qkv_init(dims: int, head: int):
    head_dim = dims // head
    scale = head_dim ** -0.5
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims, bias=False)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    return q, k, v, o, scale

def create_qkv(q, k, v, x, xa=None, head=8):
    head_dim = q.out_features // head
    scale = head_dim ** -0.5
    q = q(x) * scale
    k = k(xa if xa is not None else x) * scale
    v = v(xa if xa is not None else x)
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
    def __init__(self, dims: int, head: int, max_iterations: int = 3, threshold: float = 0.01, s_factor: float = 0.1, dropout: float = 0.1):
        super(attention, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_iterations = max_iterations
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.s_factor = nn.Parameter(torch.tensor(s_factor))
        self.dropout = dropout
        
        self.q = nn.Linear(dims, dims)
        self.k = nn.Linear(dims, dims, bias=False)
        self.v = nn.Linear(dims, dims)
        self.o = nn.Linear(dims, dims)

        self.lna = nn.LayerNorm(dims, bias=False)
        self.lnb = nn.LayerNorm(dims, bias=False)      
        self.lnc = nn.LayerNorm(self.head_dim, bias=False)
        self.lnd = nn.LayerNorm(self.head_dim, bias=False)     

        self.attn_local = LocalAttentionModule(self.head_dim)

    def _focus(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        q = self.q(self.lna(x))
        k = self.k(self.lnb(x if xa is None else xa))
        v = self.v(self.lnb(x if xa is None else xa))
        
        query = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        key = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        value = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        iteration = 0
        prev_attn_out = torch.zeros_like(query)
        attn_out = torch.zeros_like(query)
        threshold = self.threshold.item()
        s_factor = self.s_factor.item()

        q_current = query

        while iteration < self.max_iterations:
            eff_span = min(x.shape[1], q_current.size(1), key.size(1))
            if xa is not None:
                eff_span = min(eff_span, xa.shape[1])

            if eff_span == 0: 
                break

            q_iter = q_current[:, :, :eff_span, :]
            k_iter = key[:, :, :eff_span, :]
            v_iter = value[:, :, :eff_span, :]

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
                q_proj, k_proj, v_proj,
                mask=iter_mask,
                is_causal=True
            )

            attn_out_span = self.attn_local._reshape_to_output(attn_output_iter)
            if attn_out_span.dim() == 4:
                b, h, s, d = attn_out_span.shape
                projected_attn_out_span = self.attn_local.out_proj(attn_out_span.view(-1, d)).view(b, h, s, -1)
            elif attn_out_span.dim() == 3:
                b, s, d = attn_out_span.shape
                if d == self.head_dim:
                    projected_attn_out_span = self.attn_local.out_proj(attn_out_span.view(-1, d)).view(b, 1, s, -1)
                elif d == self.head * self.head_dim:
                    projected_attn_out_span = attn_out_span.view(b, self.head, s, self.head_dim)
                else:
                    raise RuntimeError(f"Cannot reshape attn_out_span of shape {attn_out_span.shape} to [b, h, s, head_dim]")
            else:
                raise RuntimeError(f"Unexpected attn_out_span shape: {attn_out_span.shape}")

            current_iter_out = torch.zeros_like(q_current)
            current_iter_out[:, :, :eff_span, :] = projected_attn_out_span

            diff = torch.abs(current_iter_out - prev_attn_out).mean()
            dynamic_threshold = threshold + s_factor * diff

            if diff < dynamic_threshold and iteration > 0:
                attn_out = current_iter_out
                break

            prev_attn_out = current_iter_out.clone()
            q_current = q_current + current_iter_out
            attn_out = current_iter_out

            iteration += 1

        output = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(output), None

    def _slide_win_local(self, x: Tensor, win_size: int, span_len: int,
                         mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        batch, ctx, dims = x.size()
        output = torch.zeros_like(x)

        num_windows = (ctx + win_size - 1) // win_size

        for i in range(num_windows):
            q_start = i * win_size
            q_end = min(q_start + win_size, ctx)
            current_window_q_len = q_end - q_start
            if current_window_q_len == 0: 
                continue

            kv_start = max(0, q_end - span_len)
            kv_end = q_end
            query_win = x[:, q_start:q_end, :]
            key_win = x[:, kv_start:kv_end, :]

            window_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    window_mask = mask[:, :, q_start:q_end, kv_start:kv_end]
                elif mask.dim() == 2:
                    window_mask = mask[q_start:q_end, kv_start:kv_end]

            attn_out_win, _ = self._focus(
                x=query_win,
                xa=key_win,
                mask=window_mask
            )

            output[:, q_start:q_end, :] = attn_out_win

        return output

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, 
                use_sliding_window: bool = False, win_size: int = 512, span_len: int = 1024) -> Tensor:
        if use_sliding_window:
            return self._slide_win_local(x, win_size, span_len, mask)
        else:
            output, _ = self._focus(x, xa, mask)
            return output

attn = attention(dims=512, head=8, max_iterations=3)

x = torch.randn(2, 100, 512)
output = attn(x)

xa = torch.randn(2, 50, 512)
output = attn(x, xa=xa)

output = attn(x, use_sliding_window=True, win_size=256, span_len=512)      
print(output)    
