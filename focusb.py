
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
from torch.nn.functional import scaled_dot_product_attention

def qkv_init(dims: int, head: int):
    head_dim = dims // head
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims, bias=False)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    lna = nn.LayerNorm(dims, bias=False)  
    lnb = nn.LayerNorm(head_dim, bias=False)
    return q, k, v, o, lna, lnb

def create_qkv(dims, head, q, k, v, x, xa):
    head_dim = dims // head
    scale = head_dim ** -0.25
    q = q(x) * scale
    k = k(xa) * scale
    v = v(xa)
    batch, ctx, dims = x.shape
    def _shape(tensor):
        return tensor.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
    return _shape(q), _shape(k), _shape(v)

def calculate_attention(q, k, v, mask=None, temp=1.0):
    scaled_q = q
    if temp != 1.0 and temp > 0:
        scaled_q = q * (1.0 / temp)**.5
        # print(temp)
    out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)        
    return out

class LocalOut(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.query_module = nn.Linear(head_dim, head_dim)
        self.key_module = nn.Linear(head_dim, head_dim)
        self.value_module = nn.Linear(head_dim, head_dim)
        self.out_proj = nn.Linear(head_dim, head_dim)
    
    def _reshape_to_output(self, x):
        return x

class attentiona(nn.Module):
    def __init__(self, dims: int, head: int, max_iter: int = 3, threshold: float = 0.01, factor: float = 0.1, dropout: float = 0.1, temp = 1.0):
        super(attentiona, self).__init__()
        self.q,  self.k,  self.v,  self.o, self.lna, self.lnb = qkv_init(dims, head)
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_iter = max_iter
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)        
        self.factor = nn.Parameter(torch.tensor(factor))
        self.lnc = nn.LayerNorm(self.head_dim, bias=False)
        self.lnd = nn.LayerNorm(self.head_dim, bias=False)     
        self.attn_local = LocalOut(self.head_dim)   

    def _focus(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        z = default(xa, x)
        q, k, v = create_qkv(self.dims, self.head, self.q, self.k, self.v, self.lna(x), self.lna(z))    

        iteration = 0
        temp = self.temp.item()
        prev_out = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        threshold = self.threshold.item()
        factor = self.factor.item()
        qcur = q

        while iteration < self.max_iter:
            eff_span = min(qcur.shape[1], k.shape[1])
            if xa is not None:
                eff_span = min(eff_span, xa.shape[1])
            if eff_span == 0: 
                break

            qiter = qcur[:, :, :eff_span, :]
            kiter = k[:, :, :eff_span, :]
            viter = v[:, :, :eff_span, :]
            q = self.attn_local.query_module(qiter)
            k = self.attn_local.key_module(kiter)
            v = self.attn_local.value_module(viter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4: 
                    iter_mask = mask[:, :, :eff_span, :eff_span]
                elif mask.dim() == 2: 
                    iter_mask = mask[:eff_span, :eff_span]

            attn_iter = calculate_attention(
                self.lnc(q), self.lnd(k), v,
                mask=iter_mask, temp=temp)

            iter_out = torch.zeros_like(qcur)
            iter_out[:, :, :eff_span, :] = attn_iter
            diff = torch.abs(iter_out - prev_out).mean()
            dthresh = threshold + factor * diff
            if diff < dthresh and iteration > 0:
                attn_out = iter_out
                break

            prev_out = iter_out.clone()
            qcur = qcur + iter_out
            attn_out = iter_out
            iteration += 1
            temp += 0.005

        output = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(output), None

    def _slide_win_local(self, x: Tensor, win_size: int, span_len: int, mask: Optional[Tensor] = None) -> Tensor:

        batch, ctx, dims = x.shape
        output = torch.zeros_like(x)
        num_win = (ctx + win_size - 1) // win_size

        for i in range(num_win):
            qstart = i * win_size
            qend = min(qstart + win_size, ctx)
            win_qlen = qend - qstart
            if win_qlen == 0: 
                continue

            kstart = max(0, qend - span_len)
            kend = qend
            qwin = x[:, qstart:qend, :]
            kwin = x[:, kstart:kend, :]

            win_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    win_mask = mask[:, :, qstart:qend, kstart:kend]
                elif mask.dim() == 2:
                    win_mask = mask[qstart:qend, kstart:kend]

            attn_out, _ = self._focus(x=qwin, xa=kwin, mask=win_mask)
            output[:, qstart:qend, :] = attn_out
        return output

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, 
                use_sliding_win: bool = False, win_size: int = 512, span_len: int = 1024) -> Tensor:
        if use_sliding_win:
            return self._slide_win_local(x, win_size, span_len, mask)
        else:
            output, _ = self._focus(x, xa, mask)
            return output
