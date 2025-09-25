# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:02:21 2022

@author: PatCa
"""

import pandas as pd
import pandas_datareader.data as web
import numpy as np

FB = web.YahooOptions('FB')

#showing available expiries
for exp in FB.expiry_dates:
     print(exp.isoformat())

# get call data
calls = FB.get_call_data()
calls

#get put data
puts  = FB.get_put_data()
puts


#get call data based on specific expiry
FB.get_call_data(month =2 , year = 2021)

# getting all call data, can also pass in a datetime to the expiry below
allcalls = FB.get_call_data(expiry= FB.expiry_dates)

#notice index is in multiindex
allcalls.index

#changing to regular index
allcalls.reset_index(inplace=True)

#get all available data for puts and calls at every expiration
alloptions = FB.get_all_data()

alloptions.reset_index(inplace=True)

# perform calculate on the data.
alloptions['mid_price'] = (alloptions.Ask - alloptions.Bid) / 2


#%%

import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

start = dt.datetime(2010,1,1)    
end =dt.datetime(2020,10,1) 
symbol = 'AAPL' ###using Apple as an example
source = 'yahoo'
data = web.DataReader(symbol, source, start, end)
data['change'] = data['Adj Close'].pct_change()
data['rolling_sigma'] = data['change'].rolling(20).std() * np.sqrt(255)


data.rolling_sigma.plot()
plt.ylabel('$\sigma$')
plt.title('AAPL Rolling Volatility')
