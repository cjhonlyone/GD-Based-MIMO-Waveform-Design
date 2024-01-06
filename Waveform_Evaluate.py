# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:08:52 2023

@author: F520-CJH
"""

import numpy as np
import matplotlib.pyplot as plt


def Calculate_CM(X):
    [N, M] = np.shape(X)
    CM = np.zeros((2*N-1, M*M))
    for i in range(M):
        for j in range(M):
            CM[:, i*M+j] = np.abs(np.correlate(X[:,i],X[:,j], "full"))/N
    CM = np.power(CM, 2)
    return CM



def Calculate_ISL_PSL(CM):
    [N, M] = np.shape(CM)
    N = (N+1)/2
    M = np.sqrt(M)
    
    Aidx = np.arange(M) * M + np.arange(M)
    Aidx = Aidx.astype(int)
    Bidx = np.concatenate((np.arange(N-1), np.arange(N,2*N-1)))
    Bidx = Bidx.astype(int)
    ACM = CM[:, Aidx]
    ACM = ACM[Bidx, :]
    Cidx = np.arange(M**2)
    Cidx = np.setdiff1d(Cidx,Aidx)
    Cidx = Cidx.astype(int)
    CCM = CM[:, Cidx]
    
    APSL = np.max(np.max(ACM))
    # AISL = np.sum(np.sum(ACM))/(2*N-2)/M
    AISL = np.sum(np.sum(ACM))
    
    CPSL = np.max(np.max(CCM)) if M > 1 else 1e-20
    # CISL = np.sum(np.sum(CCM))/(2*N-2)/(M**2-M)
    CISL = np.sum(np.sum(CCM)) if M > 1 else 1e-20
    
    PSL = np.max([APSL, CPSL])
    ISL = AISL+CISL
    
    return 10*np.log10([ISL, PSL, AISL, APSL, CISL, CPSL])


def Plot_ACM(CM):
    [N, M] = np.shape(CM)
    N = (N+1)/2
    M = np.sqrt(M)
    
    Aidx = np.arange(M) * M + np.arange(M)
    Aidx = Aidx.astype(int)
    Bidx = np.concatenate((np.arange(N-1), np.arange(N,2*N-1)))
    Bidx = Bidx.astype(int)
    ACM = CM[:, Aidx]
    ACM = ACM[Bidx, :]
    Cidx = np.arange(M**2)
    Cidx = np.setdiff1d(Cidx,Aidx)
    Cidx = Cidx.astype(int)
    CCM = CM[:, Cidx]
    
    for i in range(len(Aidx)):
        plt.plot(10*np.log10(CM[:,Aidx[i]]))
        
    plt.ylim([-50, 1]);
    
    plt.show()
    
    
def Plot_CCM(CM):
    [N, M] = np.shape(CM)
    N = (N+1)/2
    M = np.sqrt(M)
    
    Aidx = np.arange(M) * M + np.arange(M)
    Aidx = Aidx.astype(int)
    Bidx = np.concatenate((np.arange(N-1), np.arange(N,2*N-1)))
    Bidx = Bidx.astype(int)
    ACM = CM[:, Aidx]
    ACM = ACM[Bidx, :]
    Cidx = np.arange(M**2)
    Cidx = np.setdiff1d(Cidx,Aidx)
    Cidx = Cidx.astype(int)
    CCM = CM[:, Cidx]
    
    for i in range(len(Cidx)):
        plt.plot(10*np.log10(CM[:,Cidx[i]]))
        
    plt.ylim([-50, 1]);
    
    plt.show()

import scipy.io as scio
if __name__ == '__main__':
    dataFile = 'pyWaveform.mat'
    data = scio.loadmat(dataFile)
    waveform = data['waveform']
    M = int(data['M'])
    N = int(data['N'])
    X = np.transpose(np.reshape(np.exp(1j*waveform*np.pi*2), [M, N]))
    cm = Calculate_CM(X)
    [ISL, PSL, AISL, APSL, CISL, CPSL] = Calculate_ISL_PSL(cm)
    Plot_ACM(cm)
    Plot_CCM(cm)
    # waveform

    # print("M = {:d}, N = {:d}, APSL = {: .2f} dB".format(M, N, APSL))
    # print("M = {:d}, N = {:d}, AISL = {: .2f} dB".format(M, N, AISL))
    # print("M = {:d}, N = {:d}, CPSL = {: .2f} dB".format(M, N, CPSL))
    # print("M = {:d}, N = {:d}, CISL = {: .2f} dB".format(M, N, CISL))
    print("M = {:d}, N = {:d},  PSL = {: .2f} dB".format(M, N, PSL))
    print("M = {:d}, N = {:d},  ISL = {: .2f} dB".format(M, N, ISL))
    print("M = {:d}, N = {:d}, Welch = {: .2f} dB".format(M, N, 10*np.log10((M-1)/(2*M*N-M-1))))





