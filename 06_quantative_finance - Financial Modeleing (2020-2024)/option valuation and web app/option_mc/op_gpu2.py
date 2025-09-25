import cupy
import numpy as np
import math
import time
import numba
from numba import cuda
from numba import njit
from numba import prange
import gc
cupy.cuda.set_allocator(None)

N_PATHS = int(8192000/4)
N_STEPS = 365
T = 1.0
K = 110.0
B = 100.0
S0 = 120.0
sigma = 0.35
mu = 0.1
r = 0.05

@cuda.jit
def numba_gpu_barrier_option(d_s, T, K, B, S0, sigma, mu, r, d_normals, N_STEPS, N_PATHS):
    # ii - overall thread index
    ii = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    tmp1 = mu*T/N_STEPS
    tmp2 = math.exp(-r*T)
    tmp3 = math.sqrt(T/N_STEPS)
    running_average = 0.0
    for i in range(ii, N_PATHS, stride):
        s_curr = S0
        for n in range(N_STEPS):
            s_curr += tmp1 * s_curr + sigma*s_curr*tmp3*d_normals[i + n * N_PATHS]
            running_average += (s_curr - running_average) / (n + 1.0)
            if running_average <= B:
                break
        payoff = running_average - K if running_average>K else 0
        d_s[i] = tmp2 * payoff
        

s = time.time()        
randoms_gpu = cupy.random.normal(0, 1, N_PATHS * N_STEPS, dtype=cupy.float32)
output =  np.zeros(N_PATHS, dtype=np.float32)

threadsperblock = 128
blockspergrid = (N_PATHS-1) // threadsperblock + 1
output = cupy.zeros(N_PATHS, dtype=cupy.float32)
numba_gpu_barrier_option[(blockspergrid,), (threadsperblock,)](output, np.float32(T), np.float32(K), 
                    np.float32(B), np.float32(S0), 
                    np.float32(sigma), np.float32(mu), 
                    np.float32(r), randoms_gpu, N_STEPS, N_PATHS)
v = output.mean()
cuda.synchronize()
e = time.time()
print('time', e-s, 'v', v)

del randoms_gpu 
del output
gc.collect()