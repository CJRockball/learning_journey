# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:35:17 2022

@author: PatCa
"""
#European option
from random import gauss
from math import exp, sqrt
import time
import datetime

S_0   = 114.64          # underlying spot price
sigma = 0.088864        # volatility of 20.00%
r     =   0.033         # risk free rate of 1.18%
T     = 2/365.0         # maturity in 100 days
K     = 100             # strike 
paths = 1000000 #int(sys.argv[1])

S_0 = 857.29 # underlying price
sigma = 0.2076 # vol of 20.76%
r = 0.0014 # rate of 0.14%
T = (datetime.date(2013,9,21) - datetime.date(2013,9,3)).days / 365.0
K = 860.

payoffs = []
discount_factor = exp(-r * T)

def generate_asset_price(S_0, sigma, r, T):
    return S_0 * exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * gauss(0,1.0))

def call_payoff(S_T, K):
    return max(S_T - K, 0.0)

s = time.time()
for i in range(paths):
    S_T = generate_asset_price(S_0, sigma, r, T)
    payoffs.append(call_payoff(S_T, K))


price = discount_factor * (sum(payoffs) / float(paths))
e = time.time() - s
print("calculation time %.8f" % e)
print ("Call price: %.4f" % price)


#%%


import numpy as np
import scipy.stats as stats

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

s = time.time()
#v = blackScholes_py(100., 110., 2., 0.2, 0.03)
#v = blackScholes_py(114.64, 110., 2., 0.088864, 0.033)
v = blackScholes_py(S_0, K, T, sigma, r)
end = time.time() - s
print("value: {}, time: {:.8f}".format(v,end))
























