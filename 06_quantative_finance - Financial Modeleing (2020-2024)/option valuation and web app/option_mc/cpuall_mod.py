# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:42:58 2022

@author: PatCa
"""

import cupy
import numpy as np
import math
import time
import numba
#from numba import cuda
from numba import njit
from numba import prange
#import cudf
#cupy.cuda.set_allocator(None)
import gc

N_PATHS = int(8192000/4)
N_STEPS = 365
T = 1.0
K = 110.0
B = 100.0
S0 = 120.0
sigma = 0.35
mu = 0.1
r = 0.05

@njit(fastmath=True, cache=True, parallel=True, nogil=True)
def cpu_multiplecore_barrier_option(d_s, T, K, B, S0, sigma, mu, r, N_STEPS, N_PATHS):
    tmp1 = mu*T/N_STEPS
    tmp2 = math.exp(-r*T)
    tmp3 = math.sqrt(T/N_STEPS)
    for i in prange(N_PATHS):
        s_curr = S0
        running_average = 0.0
        randoms_cpu = np.random.normal(0,1,N_STEPS)
        for n in range(N_STEPS):
            s_curr += tmp1 * s_curr + sigma*s_curr*tmp3*randoms_cpu[n]
            running_average = running_average + 1.0/(n + 1.0) * (s_curr - running_average)
            if running_average <= B:
                break
        payoff = running_average - K if running_average>K else 0
        d_s[i] = tmp2 * payoff
    return d_s
        


output =  np.zeros(N_PATHS, dtype=np.float32)
s = time.time()
out = cpu_multiplecore_barrier_option(output, np.float32(T), np.float32(K), 
                    np.float32(B), np.float32(S0), 
                    np.float32(sigma), np.float32(mu), 
                    np.float32(r), np.int32(N_STEPS), np.int32(N_PATHS))
v = out.mean()
e = time.time()
print('time', e-s, 'v', v)

del output
del out
gc.collect()
