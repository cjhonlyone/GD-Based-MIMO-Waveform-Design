# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:34:39 2023

@author: F520-CJH
"""

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch


class NNGDLOSS(nn.Module):
    def __init__(self, M=3, N=4, t=1, G=50, E=100, W=0.1, device=torch.device("cuda:0"), dtype=torch.float32, name="NNGDLOSS"):
        super(NNGDLOSS, self).__init__()
        self.M = M
        self.N = N
        self.t = t
        self.G = G
        self.W = W
        self.E = E

        self.Groupdiag_cm = self.W * torch.ones([self.M * self.M, self.N], device=device, dtype=dtype)
        # self.Groupdiag_cm[:,0:1] = (1-self.W)*torch.ones([self.M * self.M, 1], device=device, dtype=dtype)
        if self.G == 0:
            self.Groupdiag_cm[:,(self.N-self.E-1+1):(self.N-self.G)] = (1-self.W)*torch.ones([int(self.M * self.M), self.E-self.G], device=device, dtype=dtype)
            self.Groupdiag_cm[:,0:1] = (1-self.W)*torch.ones([self.M * self.M, 1], device=device, dtype=dtype)
        else:
            self.Groupdiag_cm[:,(self.N-self.E-1+1):(self.N-self.G+1)] = (1-self.W)*torch.ones([int(self.M * self.M), self.E-self.G + 1], device=device, dtype=dtype)
            self.Groupdiag_cm[:,0:1] = self.W * torch.ones([self.M * self.M, 1], device=device, dtype=dtype)

    def forward(self, y_pred, y_true):
        loss, PSL, APSL, CPSL, ISL, AISL, CISL = self.Ewaveform2losshalf(y_pred[0])
        # return [loss, 10*torch.log10(PSL), 10*torch.log10(APSL), 10*torch.log10(CPSL), 10*torch.log10(ISL), 10*torch.log10(AISL), 10*torch.log10(CISL)]
        return [loss, PSL, APSL, CPSL, ISL, AISL, CISL]

    def Ewaveform2losshalf(self, x):
        device = x.device
        dtype = x.dtype
        x = x[0:(self.M * self.N)]
        yrad = 2 * np.pi * x
        y = torch.transpose(torch.reshape(yrad, [self.M, self.N]), 0, 1)
        y_ap = torch.cat([torch.zeros([self.N - 1, self.M], device=device, dtype=dtype), y], 0)
        y_p = torch.cat([y[1:, :], y], 0)
        y_ba = y_ap

        y_s = torch.exp(1j * y);

        cm = torch.zeros([self.M * self.M, self.N], device=device, dtype=dtype)
        y_s_spec = torch.transpose(torch.fft.fft(y_s, 2 * self.N, dim=0), 0, 1)
        y_s_spec_re1 = y_s_spec.repeat(self.M, 1)
        y_s_spec_re2 = torch.reshape(y_s_spec.repeat(1, self.M), [self.M * self.M, 2 * self.N])
        sigma_sss = torch.fft.ifft(y_s_spec_re1 * torch.conj(y_s_spec_re2), dim=1)
        cm[:, 0] = torch.square(sigma_sss[:, 0].abs())
        cm[:, 1:] = torch.square(sigma_sss[:, self.N+1:].abs())

        cm = cm * self.Groupdiag_cm

        idx_eye = torch.tensor([i * self.M + i for i in range(self.M)], device=device)
        idx_oth = torch.reshape(
            torch.tensor([[i * self.M + j for j in range(self.M) if j != i] for i in range(self.M)], device=device),
            [-1])

        A = cm.index_select(0, idx_eye).t()
        # A_N = torch.cat([A[0:(self.N - 1), :], A[self.N:, :]], 0)
        A_N = A[1:]
        E = cm.index_select(0, idx_oth).t()

        APSL = torch.reshape(A_N, (-1,))
        CPSL = torch.reshape(E, (-1,))

        ac = torch.cat((APSL, CPSL), 0) / self.N / self.N if self.M > 1 else APSL / self.N / self.N
        # loss = torch.logsumexp(ac, 0)
        loss = self.t * torch.logsumexp(ac / self.t, 0)
        acd = ac.detach()
        APSLd = APSL.detach()
        CPSLd = CPSL.detach()
        PSL_o = torch.max(acd)
        APSL_o = torch.max(APSLd)
        CPSL_o = torch.max(CPSLd) if self.M > 1 else 1e-20
        ISL_o = torch.sum(acd)
        AISL_o = torch.sum(APSLd)
        CISL_o = torch.sum(CPSLd) if self.M > 1 else 1e-20
        return [loss, PSL_o, APSL_o, CPSL_o, ISL_o, AISL_o, CISL_o]

if __name__ == '__main__':
    dtype = torch.float
    # device = torch.device("cpu")
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    M = 8
    N = 32
    model = EOF(M=M, N=N, device=device)
    # waveform = torch.linspace(0.01, M * N * 0.01, M * N, device=device, dtype=dtype)
    waveform = torch.randn(1, M * N, device=device, dtype=dtype)
    loss = model.Ewaveform2losshalf(waveform)
    print(loss)
