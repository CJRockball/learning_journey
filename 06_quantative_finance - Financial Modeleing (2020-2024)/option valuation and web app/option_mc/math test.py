# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:01:12 2022

@author: PatCa
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(x)

def forward_difference(f, x, dx):
    return(f(x+dx) - f(x))/dx

delta_x = [1.5, 1, 0.5, 0.2, 0.1]
nums = np.arange(1,5,0.01)
true = f(nums)

plt.figure()
plt.plot(nums, true, label='True', linewidth=3)
for delt in delta_x:
    plt.plot(nums, forward_difference(f, nums, delt),label=f'$\Delta x$ {delt}',linewidth=2, linestyle='--')

plt.legend()
plt.ylabel('Derivative')
plt.xlabel('$x$')
plt.title("Forward Finite Difference")

def backward_difference(f, x, dx):
    return (f(x) - f(x-dx)) / dx

plt.figure()
plt.plot(nums, true, label= 'True',linewidth=3)
for delt in delta_x:
    plt.plot(nums, backward_difference(f, nums, delt),
             label=f'$\Delta x$ = {delt}',
             linewidth=2, linestyle='--')

plt.legend()
plt.ylabel("Derivative")
plt.xlabel('$x$')
plt.title('Backward Finite Difference')

def central_difference(f, x, dx):
    return (f(x+dx) - f(x-dx)) / (2*dx)

plt.figure()
plt.plot(nums, true, label= 'True',linewidth=3)
for delt in delta_x:
    plt.plot(nums, central_difference(f, nums, delt), label=f'$\Delta x$ = {delt}',linewidth=2,linestyle='--')

plt.legend()
plt.ylabel("Derivative")
plt.xlabel('$x$')
plt.title('Central Finite Difference')



























