import cupy
import numpy as np
import math
import time
import numba
from numba import cuda
from numba import njit
from numba import prange
#import cudf
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

s = time.time()
randoms_gpu = cupy.random.normal(0, 1, N_PATHS * N_STEPS, dtype=cupy.float32)
randoms_cpu = np_randoms = cupy.asnumpy(randoms_gpu)
#output =  np.zeros(N_PATHS, dtype=np.float32)




cupy_barrier_option = cupy.RawKernel(r'''
extern "C" __global__ void barrier_option(
    float *d_s,
    const float T,
    const float K,
    const float B,
    const float S0,
    const float sigma,
    const float mu,
    const float r,
    const float * d_normals,
    const long N_STEPS,
    const long N_PATHS)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned tid = threadIdx.x;

  const float tmp1 = mu*T/N_STEPS;
  const float tmp2 = exp(-r*T);
  const float tmp3 = sqrt(T/N_STEPS);
  double running_average = 0.0;

  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    float s_curr = S0;
    unsigned n=0;
    for(unsigned n = 0; n < N_STEPS; n++){
       s_curr += tmp1 * s_curr + sigma*s_curr*tmp3*d_normals[i + n * N_PATHS];
       running_average += (s_curr - running_average) / (n + 1.0) ;
       if (running_average <= B){
           break;
       }
    }

    float payoff = (running_average>K ? running_average-K : 0.f);
    d_s[i] = tmp2 * payoff;
  }
}

''', 'barrier_option')


number_of_threads = 256
number_of_blocks = (N_PATHS-1) // number_of_threads + 1
output = cupy.zeros(N_PATHS, dtype=cupy.float32)

cupy_barrier_option((number_of_blocks,), (number_of_threads,),
                   (output, np.float32(T), np.float32(K), 
                    np.float32(B), np.float32(S0), 
                    np.float32(sigma), np.float32(mu), 
                    np.float32(r),  randoms_gpu, N_STEPS, N_PATHS))
v = output.mean()
cupy.cuda.stream.get_current_stream().synchronize()
e = time.time()
print('time', e-s, 'v',v)




