# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:07:31 2022

@author: PatCa
"""

import numpy as np
from scipy.stats import norm

N = norm.cdf
N_prime = norm.pdf

# Price Formula
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

# Delta Formula
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + sigma**2/2)*T) /\
                     sigma*np.sqrt(T)

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma* np.sqrt(T)

def delta_call(S, K, T, r, sigma):
    N = norm.cdf
    return N(d1(S, K, T, r, sigma))

def delta_put(S, K, T, r, sigma):
    return - N(-d1(S, K, T, r, sigma))


#Gamma Formula
def gamma(S, K, T, r, sigma): 
    return N_prime(d1(S,K, T, r, sigma))/(S*sigma*np.sqrt(T))


# Vega
def vega(S, K, T, r, sigma):
    return S*np.sqrt(T)*N_prime(d1(S,K,T,r,sigma)) 


#Theta
def theta_call(S, K, T, r, sigma):
    p1 = - S*N_prime(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(d2(S, K, T, r, sigma)) 
    return p1 - p2

def theta_put(S, K, T, r, sigma):
    p1 = - S*N_prime(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(-d2(S, K, T, r, sigma)) 
    return p1 + p2


#Rho
def rho_call(S, K, T, r, sigma):
    return K*T*np.exp(-r*T)*N(d2(S, K, T, r, sigma))

def rho_put(S, K, T, r, sigma):
    return -K*T*np.exp(-r*T)*N(-d2(S, K, T, r, sigma))



