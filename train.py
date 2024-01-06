# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:43:36 2023

@author: F520-CJH
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import scipy.io as scio
import Waveform_Evaluate as wf

import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

from torchviz import make_dot, make_dot_from_trace

from thop import profile

from NNGD import *
from NNGDLOSS import *
from lrscheduler import *

def TrainNN(job):

    logdir = "./log/"
    modeldir = "./model/"
    paramdir = "./param/"

    mybatch_size = 1
    mybatch_length = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    epochs = job['Epochs']

    # job['t'] = job['t']/job['N']/job['N']/np.log(job['N']*job['M'])
    # job['t'] = (job['M']-1)/(2*job['N']*job['M']-job['M']-1)/(np.log(job['N']*job['M']))

    learning_rate = 0.01
    model = job['Struct'](M=job['M'], N=job['N'], num_blocks=job['Layer'])
    loss_fn = job['Loss'](M=job['M'], N=job['N'], t = job['t'], G = job['G'], E = job['E'], W = job['W'], device=device, dtype=dtype)
    x = torch.rand([mybatch_size*mybatch_length, job['M']*job['N']], device=device, dtype=dtype)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    LearningRateScheduler = job["Learning Rate"]
    job["Params Count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)

    history = {"loss":[], "lr":[], "PSLTrace":[], "ISLTrace":[], "APSLTrace":[], "AISLTrace":[], "CPSLTrace":[], "CISLTrace":[]}
    # model.train()
    PSL_old = 0
    for t in range(epochs):
        pred = model(x)
        [loss, PSL, APSL, CPSL, ISL, AISL, CISL] = loss_fn(pred, pred)
        history["loss"].append(loss)
        history["PSLTrace"].append(PSL)
        history["ISLTrace"].append(ISL)
        history["APSLTrace"].append(APSL)
        history["AISLTrace"].append(AISL)
        history["CPSLTrace"].append(CPSL)
        history["CISLTrace"].append(CISL)

        # if torch.abs(PSL-PSL_old)/PSL_old <= 1e-8:
        #     job['Epochs'] = t
        #     break;

        # PSL_old = PSL

        # Backpropagation
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        for p in optimizer.param_groups:
            history["lr"].append(p['lr'])
            p['lr'] = LearningRateScheduler(t, p['lr'])
            if t % 100 == 99:
                print(f"Epoch {t+1} loss: {loss:.2f} lr: {p['lr']:.8f} PSL { 10*torch.log10(PSL):.2f} ISL {10*torch.log10(ISL):.2f}")

    historymat = {"loss": [], "lr": [], "PSLTrace": [], "ISLTrace": [], "APSLTrace": [], "AISLTrace": [], "CPSLTrace": [], "CISLTrace": []}
    for k, v in history.items():
        historymat[k] = np.array(torch.tensor(v).numpy(), dtype=np.double)
    scio.savemat('trainloss.mat', historymat)

    with torch.no_grad():
        # model.eval()
        waveform = model(x).cpu().numpy()
        waveform = waveform[:,0:(job['M']*job['N'])]
    return waveform

def CalSL(job, waveform):

    # logdir = "./log/"
    # modeldir = "./model/"
    # paramdir = "./param/"
    
    # x = tf.random.uniform([1, job['M']*job['N']],0,1)
    
    
    # loaded = tf.saved_model.load(modeldir)
    # # print(list(loaded.signatures.keys()))
    # infer = loaded.signatures["serving_default"]
    # # print(infer.structured_outputs)
    # waveform = infer(x)
    # waveform = waveform['output_1'].numpy()

    LogSL(job, waveform)
    

def LogSL(job, waveform):
    X = np.transpose(np.reshape(np.exp(1j*waveform*np.pi*2), [job['M'], job['N']]))
    cm = wf.Calculate_CM(X)

    Groupdiag_cm = np.ones([2*job["N"] - 1, job["M"] * job["M"]]) * 1e-50
    Groupdiag_cm[(job["N"]+job["G"]-1):(job["N"]+job["E"]), :] = np.ones([job["E"]-job["G"]+1,job["M"] * job["M"]])
    Groupdiag_cm[(job["N"]-job["E"]-1):(job["N"]-job["G"]), :] = np.ones([job["E"]-job["G"]+1,job["M"] * job["M"]])
    [ISL, PSL, AISL, APSL, CISL, CPSL] = wf.Calculate_ISL_PSL(cm * Groupdiag_cm)

    Groupdiag_cm = np.ones([2*job["N"] - 1, job["M"] * job["M"]]) 
    Groupdiag_cm[(job["N"]+job["G"]-1):(job["N"]+job["E"]), :] = np.ones([job["E"]-job["G"]+1,job["M"] * job["M"]]) * 1e-50
    Groupdiag_cm[(job["N"]-job["E"]-1):(job["N"]-job["G"]), :] = np.ones([job["E"]-job["G"]+1,job["M"] * job["M"]]) * 1e-50
    [ISLo, PSLo, AISLo, APSLo, CISLo, CPSLo] = wf.Calculate_ISL_PSL(cm * Groupdiag_cm)
    # wf.Plot_ACM(cm);
    # wf.Plot_CCM(cm);
    
    # write to csv
    
    print("M = {:d}, N = {:d},  PSL = {: .2f} dB, ISL = {: .2f} dB, APSL = {: .2f} dB, AISL = {: .2f} dB, CPSL = {: .2f} dB, CISL = {: .2f} dB".format(job['M'], job['N'], PSL, ISL, APSL, AISL, CPSL, CISL))
    print("M = {:d}, N = {:d},  PSL = {: .2f} dB, ISL = {: .2f} dB, APSL = {: .2f} dB, AISL = {: .2f} dB, CPSL = {: .2f} dB, CISL = {: .2f} dB".format(job['M'], job['N'], PSLo, ISLo, APSLo, AISLo, CPSLo, CISLo))
    print("M = {:d}, N = {:d}, WelchPSL = {: .2f} dB".format(job['M'], job['N'], 10*np.log10((job['M']-1)/(2*job['M']*job['N']-job['M']-1))))
    print("M = {:d}, N = {:d}, WelchISL = {: .2f} dB".format(job['M'], job['N'], 10*np.log10(job['M']*(job['M']-1))))
    
    scio.savemat('pyWaveform.mat', {'waveform':np.array(waveform, dtype=np.double),
                                              "M":job['M'],
                                              "N":job['N'],
                                                "PSL":PSL,
                                                "ISL":ISL,
                                                "APSL":APSL,
                                                "AISL":AISL,
                                                "CPSL":CPSL,
                                                "CISL":CISL,
                                                "PSLo":PSLo,
                                                "ISLo":ISLo,
                                                "APSLo":APSLo,
                                                "AISLo":AISLo,
                                                "CPSLo":CPSLo,
                                                "CISLo":CISLo,
                                                "G":job["G"],
                                                "E":job["E"],
                                                "W":job["W"]
                                              })
    job["Result"] = {"PSL":PSL,"ISL":ISL,"APSL":APSL,"AISL":AISL,"CPSL":CPSL,"CISL":CISL}