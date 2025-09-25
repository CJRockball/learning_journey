# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:43:51 2022

@author: PatCa
"""

import cupy
import numpy as np
import math
from numba import njit
from numba.pycc import CC

cc = CC("my_module")


@cc.export("bar_option", 'f8[:](f8[:],f8,f8,f8,f8,f8,f8,f8,f8[:],i2,i2)')
def cpu_barrier_option(d_s, T, K, B, S0, sigma, mu, r, d_normals, N_STEPS, N_PATHS):
    tmp1 = mu*T/N_STEPS
    tmp2 = math.exp(-r*T)
    tmp3 = math.sqrt(T/N_STEPS)
    running_average = 0.0
    for i in range(N_PATHS):
        s_curr = S0
        for n in range(N_STEPS):
            s_curr += tmp1 * s_curr + sigma*s_curr*tmp3*d_normals[i + n * N_PATHS]
            running_average = running_average + 1.0/(n + 1.0) * (s_curr - running_average)
            if running_average <= B:
                break

        payoff = running_average - K if running_average>K else 0
        d_s[i] = tmp2 * payoff
    return d_s

if __name__ == "__main()__":
    cc.compile()



















 