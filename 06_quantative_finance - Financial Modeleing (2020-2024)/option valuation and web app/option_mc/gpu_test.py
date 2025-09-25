#%%

import cupy as cp
import numpy as np

#%%

array_cpu = np.random.randint(0,255,size=(2000,2000))
print(array_cpu.nbytes / 1e6)

#copy to gpu add wrapper (still in gpu)
array_gpu = cp.asarray(array_cpu)
print(type(array_gpu))

#array_gpu.copy_to_host() will actually move it back

# %%
from scipy import fft
import time

s = time.time()
a = fft.fftn(array_cpu)
e = time.time() - s
print(f'Time for CPU transform {e*1e3}ms')

from cupyx.scipy import fft as fft_gpu
import gc

time_list = []
b = 0

for i in range(1000):
    s = time.time()
    b = fft_gpu.fftn(array_gpu)
    e = time.time() - s
    time_list.append(e)

time_arr = np.array(time_list)
time_mean = np.mean(time_list)*1e3
print(f'Time for GPU transform {time_mean}ms')

gc.collect()

# %%

#send back to cpu
c = cp.asnumpy(b)


# %% NUMBA
# Add 1 to each element in an array
from numba import cuda

@cuda.jit
def add_one_kernel(A):
    #Get rows and cols
    row,column = cuda.grid(2)
    if row < A.shape[0] and column < A.shape[1]:
        #CUda kernels can't return things only operate on existing data
        A[row,column] += 1

#%% Blocks and grids

# Coord relative to block
#Find position of block
bx = cuda.blockIdx.x
by = cuda.blockIdx.y

# shape of block
bw = cuda.blockDim.x
bh = cuda.blockDim.y
print(f'bw: {bw}')
print(f'bh: {bh}')

#Thread coord
tx = cuda.threadIdx.x
ty = cuda.threadIdx.y

#Absolute coords
x = bw*bx +tx
y = bh*by + ty
#or
x,y = cuda.grid(2)

# Check how many blocks
bpg = cuda.gridDim.x

#%%%
#Matrix multiplication example

#cuda.jit
def matmul(A,B,C):
    i,j = cuda.grid(2)
    if i<C.shape[0] and j< C.shape[1]:
        tmp =0
        for k in range(A.shape[1]):
            tmp += A[i,k] * B[k,j]
        C[i,j]
        
#matmul[blockspergrid, threadsperblock](A,B,C)

#%% Matrix multiplication using VRAM
TPB = 2
@cuda.jit()
def fast_matmul(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)
    sB = cuda.shared.array(shape=(TPB,TPB), dtype=np.float32)
    x,y = cuda.grid()
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # Get num of blocks
    bpg = cuda.gridDim.x
    
    if x>= C.shape[0] and y >= C.shape[1]:
        return
    
    tmp=0
    for i in range(bpg):
        sA[tx,ty] = A[x,ty + i*TPB]
        sB[tx,ty] = B[tx + i*TPB,y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx,j]*sB[j,ty]
        cuda.syncthreads()            
    C[x,y]


























