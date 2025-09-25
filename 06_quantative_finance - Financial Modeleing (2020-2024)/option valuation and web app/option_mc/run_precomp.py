# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:54:07 2022

@author: PatCa
"""

import cupy
import numpy as np
import math
import time
from numba import njit
import my_module

if 'randoms_cpu' in vars() or 'random_cpu' in globals():
    del randoms_cpu
    del output  

N_PATHS = int(8192000/4)
N_STEPS = 365
T = 1.0
K = 110.0
B = 100.0
S0 = 120.0
sigma = 0.35
mu = 0.1
r = 0.05

randoms_cpu = np.random.normal(0,1,N_PATHS*N_STEPS)
output =  np.zeros(N_PATHS, dtype=np.float32)

s = time.time()
output1 = my_module.bar_option(output, np.float32(T), np.float32(K), 
                    np.float32(B), np.float32(S0), 
                    np.float32(sigma), np.float32(mu), 
                    np.float32(r), randoms_cpu, np.int32(N_STEPS), np.int32(N_PATHS))
v = output.mean()
e = time.time()
print('time', e-s, 'v', v)