from BS_pricing import bs_call, bs_put 
import numpy as np
import matplotlib.pyplot as plt


K = 100
r = 0.1
T = 1
sigma = 0.3

S = np.arange(60,140, 1)

calls = [bs_call(s, K, T, r, sigma) for s in S]
put = [bs_put(s, K, T, r, sigma) for s in S]

plt.figure()
plt.plot(S, calls, label="Call Value")
plt.plot(S, put, label="Put Value")
plt.xlabel("$S_0$")
plt.ylabel("Value")
plt.legend()
plt.show()
