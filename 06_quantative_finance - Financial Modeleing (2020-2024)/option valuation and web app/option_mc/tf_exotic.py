# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 09:23:26 2022

@author: PatCa
"""

import numpy as np
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%%

	
def blackScholes_py(S_0, strike, time_to_expiry, implied_vol, riskfree_rate):
    S = S_0
    K = strike
    dt = time_to_expiry
    sigma = implied_vol
    r = riskfree_rate
    Phi = stats.norm.cdf
    d_1 = (np.log(S_0 / K) + (r+sigma**2/2)*dt) / (sigma*np.sqrt(dt))
    d_2 = d_1 - sigma*np.sqrt(dt)
    return S*Phi(d_1) - K*np.exp(-r*dt)*Phi(d_2)
 
blackScholes_py(100., 110., 2., 0.2, 0.03)


#%%
