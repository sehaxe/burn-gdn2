#!/usr/bin/env python3
"""Compare burn-gdn2 (CUDA) against an equivalent pure-PyTorch GDN-2 layer.

The torch reference replicates lit_gpt/gdn2.py's fused-recurrent math with
plain torch ops (no Triton), on the same GPU, same configs.
Run burn side: cargo test --release --features cuda -p burn-gdn2 --test bench_cuda -- --ignored --nocapture
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Gdn2Torch(nn.Module):
    def __init__(self, d, h, hk, ev=1.0, kv=None):
        super().__init__()
        hv = kv or h
        kd, vh, vd = h * hk, int(hk * ev), (kv or h) * int(hk * ev)
        self.d, self.h, self.hk, self.hv, self.vh = d, h, hk, hv, vh
        self.kd, self.vd = kd, vd
        gain = 2 ** -2.5
        lin = lambda i, o: (nn.Linear(i, o, bias=False).weight.data.uniform_(-gain * (6 / (i + o)) ** 0.5, gain * (6 / (i + o)) ** 0.5), o)
        self.q_w = nn.Parameter(torch.randn(kd, d) * 0.05)
        self.k_w = nn.Parameter(torch.randn(kd, d) * 0.05)
        self.v_w = nn.Parameter(torch.randn(vd, d) * 0.05)
        self.f_w0 = nn.Parameter(torch.randn(vh, d) * 0.05)
        self.f_w1 = nn.Parameter(torch.randn(kd, vh) * 0.05)
        self.b_w = nn.Parameter(torch.randn(kd, d) * 0.05)
        self.w_w = nn.Parameter(torch.randn(vd, d) * 0.05)
        self.g_w0 = nn.Parameter(torch.randn(vh, d) * 0.05)
        self.g_w1 = nn.Parameter(torch.randn(vd, vh) * 0.05)
        self.g_b1 = nn.Parameter(torch.zeros(vd))
        self.A_log = nn.Parameter(torch.log(torch.rand(h) * 15 + 1))
        dt = torch.exp(torch.rand(kd) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001))) + torch.log(torch.tensor(0.001))).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm_w = nn.Parameter(torch.ones(vh))
        self.o_w = nn.Parameter(torch.randn(d, vd) * 0.05)
        self.q_cw = nn.Parameter(torch.randn(kd, 4) * 0.5)
        self.k_cw = nn.Parameter(torch.randn(kd, 4) * 0.5)
        self.v_cw = nn.Parameter(torch.randn(vd, 4) * 0.5)

    def short_conv(self, x, w):
        x_pad = torch.cat([x[:, :1].repeat(1, 3, 1), x], dim=1)
        out = None
        for i in range(4):
            y = x_pad[:, i:i + x.size(1)] * w[:, i]
            out = y if out is None else out + y
        return F.silu(out)

    def forward(self, x):
        B, T, _ = x.shape
        H, HK, HV, VH = self.h, self.hk, self.hv, self.vh
        KD, VD = self.kd, self.vd
        q = self.short_conv(x @ self.q_w.t(), self.q_cw)
        k = self.short_conv(x @ self.k_w.t(), self.k_cw)
        v = self.short_conv(x @ self.v_w.t(), self.v_cw)
        g = -self.A_log.exp().repeat_interleave(HK) * F.softplus(x @ self.f_w0.t() @ self.f_w1.t() + self.dt_bias)
        b = (x @ self.b_w.t()).sigmoid()
        w_gate = (x @ self.w_w.t()).sigmoid()
        q = q.reshape(B, T, H, HK).transpose(1, 2)
        k = k.reshape(B, T, H, HK).transpose(1, 2)
        g = g.reshape(B, T, H, HK).transpose(1, 2)
        v = v.reshape(B, T, HV, VH).transpose(1, 2)
        b = b.reshape(B, T, H, HK).transpose(1, 2)
        w_gate = w_gate.reshape(B, T, HV, VH).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = HK ** -0.5
        S = torch.zeros(B, HV, HK, VH, device=x.device)
        outs = []
        for t in range(T):
            S = S * g[:, :, t].exp().unsqueeze(-1)
            bk = b[:, :, t] * k[:, :, t]
            erased = (S * bk.unsqueeze(-1)).sum(dim=2)
            v_new = w_gate[:, :, t] * v[:, :, t] - erased
            S = S + k[:, :, t].unsqueeze(-1) * v_new.unsqueeze(-2)
            outs.append((S * q[:, :, t].unsqueeze(-1)).sum(dim=2) * scale)
        o = torch.stack(outs, dim=2).transpose(1, 2)
        gate = (x @ self.g_w0.t() @ self.g_w1.t() + self.g_b1).reshape(B, T, HV, VH)
        rms = torch.sqrt(o.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        o = o / rms * self.o_norm_w * F.silu(gate)
        return o.reshape(B, T, VD) @ self.o_w.t()


def bench(label, m, x, iters, warmup=3):
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
    T = x.shape[1]
    print(f"{label:<28} {dt*1e3:>9.1f} ms  {T/dt:>12,.0f} tok/s")


def main():
    dev = "cuda"
    for d, h, hk, T, iters in [
        (256, 4, 64, 1024, 5),
        (512, 8, 64, 2048, 5),
        (1024, 8, 128, 4096, 2),
    ]:
        m = Gdn2Torch(d, h, hk).to(dev).eval()
        x = torch.randn(1, T, d, device=dev)
        print(f"--- d={d} h={h} hk={hk} T={T} ---")
        bench("torch fused (token scan)", m, x, iters)


if __name__ == "__main__":
    main()
