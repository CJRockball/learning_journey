#%%import numpy as np
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az


#df = web.DataReader('GE', 'quandl', '2019-09-10', '2019-10-09')
data =yf.download("AAPL", start="2017-01-01", end="2022-03-17")
print(data.shape)
print(data.columns)
print(data.head)

# %%

close_data = data['Close']
returns = np.log(close_data[1:]/close_data[0:-1].values)
close_date = data.index

plt.figure()
plt.subplot(1,2,1)
plt.plot(close_date, close_data)
plt.subplot(1,2,2)
plt.plot(close_date[1:], returns)
plt.show()


# %%
s = 90
n = len(returns) - s
returns1 = returns[:-s]


with pm.Model() as model:
    sigma = pm.Exponential('sigma', 1./0.02, testval=0.2)
    mu = pm.Normal('mu', 0,5,testval=0.1)
    nu = pm.Exponential('nu', 1./10)
    
    logs = pm.GaussianRandomWalk('logs', sigma=sigma, shape=n)
    
    #vol_process = pm.Deterministic('vol_process', pm.math.exp(-2*logs)**0.5)
    r = pm.StudentT('r', nu, mu=mu, lam=1/np.exp(-2*logs), observed=returns1)
    
#    step = pm.NUTS()
#    trace=pm.sample(2000, tune=1000, step=step,chains=2,progressbar=True, cores=1)
    mean_field = pm.fit(20000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
    trace = mean_field.sample(3000)
#%%

# import scipy as sp

# with model:
#     start = pm.find_MAP(vars=[logs]) #, method=sp.optimize.fmin_l_bfgs_b)
    
# with model:
#     step = pm.NUTS(vars=[logs,mu,nu,sigma],scaling=start,gamma=0.25)
#     start2 = pm.sample(100, step, start=start, cores=1)[-1]
    
#     step=pm.NUTS(vars=[logs,mu,nu,sigma],scaling=start2,gamma=0.55)
#     trace = pm.sample(2000,step, start=start2,chains=4, cores=1)

#%%

pm.traceplot(trace)

#%%

gv = pm.model_to_graphviz(model)
gv.render(filename='model', format='png')


#%%

import scipy.stats as stats
test = returns[s:]

def generate_proj_returns(burn_in, trace, len_to_train):
    num_pred = 1000
    mod_returns = np.ones(shape = (num_pred,len_to_train))
    vol = np.ones(shape = (num_pred,len_to_train))
    for k in range(num_pred):
        nu = trace[burn_in+k]['nu']
        mu = trace[burn_in+k]['mu']
        sigma = trace[burn_in+k]['sigma']
        s = trace[burn_in+k]['logs'][-1]        
        for j in range(len_to_train):
            cur_log_return, s = _generate_proj_returns( mu,
                                                        s, 
                                                        nu,
                                                        sigma)
            mod_returns[k,j] = cur_log_return
            vol[k,j] = s
    return mod_returns, vol
        
def _generate_proj_returns(mu,volatility, nu, sig):
    next_vol = np.random.normal(volatility, scale=sig) #sig is SD
    
    # Not 1/np.exp(-2*next_vol), scale treated differently in scipy than pymc3
    log_return = stats.t.rvs(nu, mu, scale=np.exp(-1*next_vol))
    return log_return, next_vol

sim_returns, vol = generate_proj_returns(1000,trace,s)

#%%
ret = returns.copy().reset_index(drop=True)

[plt.plot(1/np.exp(trace[k]['logs']),color='r',alpha=.2) for k in range(1000,len(trace))]
plt.plot(ret)
[plt.plot(1+n+np.arange(0,s),1/np.exp(vol[j,:]), alpha=.01, color='y') for j in range(0,1000)]
ax = plt.gca()
ax.set_ylim([-.05,.1])


#%%

#Convert simulated returns to log-price 
prices = np.copy(sim_returns)
for k in range(len(prices)):
    cur = np.log(close_data[n])
    for j in range(len(prices[k])):
        cur = cur + prices[k, j]
        prices[k, j] = cur


#%%
training_start = 0
slope = trace[1000:]['mu'].mean()

trend = np.arange(0,s)*slope

ind = np.arange(n+1,1+n+s)
ind2 = np.arange(training_start,1+n+s)

plt.figure()
[plt.plot(ind, prices[j,:], alpha=.02, color='r') for j in range(1000)]
plt.plot(ind, trend + np.log(close_data)[n+1],
          alpha=1, 
          linewidth = 2, 
          color='black', 
          linestyle='dotted')

plt.plot(ind2,np.log(close_data)[ind2])
plt.ylim(0,5.5)
plt.xlim(trainin_start,1300)
plt.show()




















