
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional
from torch.nn.functional import scaled_dot_product_attention

def shape(dims, head, q, k, v):
    head_dim = dims // head
    scale = head_dim ** -0.25
    q = q * scale
    k = k * scale
    v = v
    def _shape(tensor):
        return tensor.view(*tensor.shape[:2], head, -1).permute(0, 2, 1, 3).contiguous()
    return _shape(q), _shape(k), _shape(v)

def qkv_init(dims: int, head: int):
    head_dim = dims // head
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims, bias=False)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    lna = nn.LayerNorm(dims, bias=False)  
    lnb = nn.LayerNorm(dims, bias=False)      
    lnc = nn.LayerNorm(head_dim, bias=False)
    lnd = nn.LayerNorm(head_dim, bias=False)    
    return q, k, v, o, lna, lnb, lnc, lnd

def calculate_attention(q, k, v, mask=None, temp=1.0):
    scaled_q = q
    if temp != 1.0 and temp > 0:
        scaled_q = q * (1.0 / temp)**.5
    out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)        
    return out

class LocalOut(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head_dim = dims // head
        self.dims = dims
        self.q_module = nn.Linear(self.head_dim, self.head_dim)
        self.k_module = nn.Linear(self.head_dim, self.head_dim)
        self.v_module = nn.Linear(self.head_dim, self.head_dim)
        self.o_proj = nn.Linear(self.head_dim, self.head_dim)

    def _reshape_to_output(self, attn_output: Tensor) -> Tensor:
        batch, _, ctx, _ = attn_output.shape
        return attn_output.transpose(1, 2).contiguous().view(batch, ctx, self.dims)    


class attentionb(nn.Module):
    def __init__(self, dims: int, head: int, max_iter: int = 3,
                 threshold: float = 0.01, factor: float = 0.1,
                 dropout: float = 0.1, temp: float = 1.0,
                 min_size: int = 64, max_size: int = 2048,
                 up_win: float = 1.1, down_win: float = 0.9,
                 up_diff: float = 0.05, down_diff: float = 0.001
        ):
        super(attentionb, self).__init__()
        self.q,  self.k,  self.v,  self.o, self.lna, self.lnb, self.lnc, self.lnd  = qkv_init(dims, head)
        self.dims = dims
        self.head = head
        self.max_iter = max_iter
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.temp_base = nn.Parameter(torch.tensor(temp), requires_grad=True)
        self.factor = nn.Parameter(torch.tensor(factor))
        self.alocal = LocalOut(dims, head)

        self.min_size = min_size
        self.max_size = max_size
        self.up_win = up_win
        self.down_win = down_win
        self.up_diff = up_diff
        self.down_diff = down_diff

    def _next_win(self, diff: float, current_size: int, max_len: int) -> int:
        new_size = current_size

        if diff > self.up_diff:
            new_size = int(current_size * self.up_win)
        elif diff < self.down_diff and current_size > self.min_size:
            new_size = int(current_size * self.down_win)
        new_size = max(self.min_size, min(new_size, max_len))
        return new_size

    def _focus(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        q = self.q(self.lna(x))
        k = self.k(self.lnb(x if xa is None else xa))
        v = self.v(self.lnb(x if xa is None else xa))
        q_full, k_full, v_full = shape(self.dims, self.head, q, k, v)

        iteration = 0
        temp = self.temp_base.item()
        prev_out = torch.zeros_like(q_full)
        attn_out = torch.zeros_like(q_full)
        threshold = self.threshold.item()
        factor = self.factor.item()
        qcur = q_full

        cur_win = min(q_full.shape[2], k_full.shape[2], self.max_size)
        cur_win = max(cur_win, self.min_size)

        while iteration < self.max_iter:
            if cur_win == 0:
                break

            qiter = qcur[:, :, :cur_win, :]
            kiter = k_full[:, :, :cur_win, :]
            viter = v_full[:, :, :cur_win, :]

            q_proj = self.alocal.q_module(qiter)
            k_proj = self.alocal.k_module(kiter)
            v_proj = self.alocal.v_module(viter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    iter_mask = mask[:, :, :cur_win, :cur_win]
                elif mask.dim() == 2:
                    iter_mask = mask[:cur_win, :cur_win]

            attn_iter = calculate_attention(
                self.lnc(q_proj), self.lnd(k_proj), v_proj,
                mask=iter_mask, temp=temp)

            iter_out = torch.zeros_like(qcur)
            iter_out[:, :, :cur_win, :] = attn_iter

            diff = torch.abs(iter_out - prev_out).mean()
            dthresh = threshold + factor * diff

            if diff < dthresh and iteration > 0:
                attn_out = iter_out
                break

            prev_out = iter_out.clone()
            qcur = qcur + iter_out
            attn_out = iter_out

            max_len = min(q_full.shape[2], k_full.shape[2])
            cur_win = self._next_win(diff, cur_win, max_len)

            iteration += 1
            temp += 0.005

        output = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(output), None

    def _slide_win_local(self, x: Tensor, win_size: int, span_len: int, mask: Optional[Tensor] = None) -> Tensor:
        _, ctx, _ = x.shape
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
