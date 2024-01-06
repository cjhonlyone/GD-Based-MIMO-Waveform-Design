# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:24:03 2023

@author: F520-CJH
"""

def lrscheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * 0.999


def lrstepscheduler(epoch, lr):
  if epoch < 2500:
    return lr
  elif epoch < 6000:
    return lr * 0.9995
  elif epoch < 8000:
    return lr * 0.9998
  elif epoch < 10000:
    return lr * 0.9999
  else:
    return lr * 0.99999


def lrholdscheduler(epoch, lr):
  return lr