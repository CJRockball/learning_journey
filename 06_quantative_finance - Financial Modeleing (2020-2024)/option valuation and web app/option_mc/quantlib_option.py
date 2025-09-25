# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:14:21 2022

@author: PatCa
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import QuantLib as ql

spot_bump = 1e-5

initial_spots_0 = np.array([100., 100., 100., 100., 100.])
initial_spots_1 = np.array([100., 100. + spot_bump, 100., 100., 100.])

corr_mat = np.matrix([[1, 0.1, -0.1, 0, 0], [0.1, 1, 0, 0, 0.2], [-0.1, 0, 1, 0, 0], [0, 0, 0, 1, 0.15], [0, 0.2, 0, 0.15, 1]])
vols = np.array([0.1, 0.12, 0.13, 0.09, 0.11])

today = ql.Date().todaysDate()
exp_date = today + ql.Period(1, ql.Years)
strike = 100
number_of_underlyings = 5

exercise = ql.EuropeanExercise(exp_date)
vanillaPayoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)

payoffAverage = ql.AverageBasketPayoff(vanillaPayoff, number_of_underlyings)
basketOptionAverage = ql.BasketOption(payoffAverage, exercise)

day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()

riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))

prices = []
for initial_spots in [initial_spots_0, initial_spots_1]:

    processes = [ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(x)), dividendTS, riskFreeTS,
                    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, y, day_count)))
                 for x, y in zip(initial_spots, vols)]

    process = ql.StochasticProcessArray(processes, corr_mat.tolist())

    rng = "pseudorandom"

    basketOptionAverage.setPricingEngine(
        ql.MCEuropeanBasketEngine(process, rng, timeStepsPerYear=1, requiredSamples=500000, seed=42) # requiredTolerance=0.01, 
    )

    prices.append(basketOptionAverage.NPV())

print((prices[1] - prices[0]) / spot_bump)