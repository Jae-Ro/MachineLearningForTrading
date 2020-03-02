"""MC1-P2: Optimize a portfolio.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
from scipy import special, optimize


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality

def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    # CHECKING FOR NaN missing Data and accounting for it with forward and backfill
    prices = prices.ffill(axis=0)
    prices = prices.bfill(axis=0)

    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    normalized_prices = prices/prices.iloc[0]

    allocs = np.ones(len(syms))  # initializing to be equal
    allocs = allocs * (1.0/float(len(syms)))

    # Setting the bounds for the optimization
    bounds = [(0, 1)]
    for i in range(len(syms)-1):
        bounds.append((0, 1))

    # FINDING THE OPTIMAL ALLOCATION SPREAD
    res = optimize.minimize(get_sharpe_ratio, args=(normalized_prices), x0=allocs,
                            bounds=(bounds), constraints=({'type': 'eq', 'fun': lambda inputs: 1-np.sum(inputs)}))

    allocs_optimized = res.x

    # Getting new portfolio values with optimized allocations
    prices_alloc = normalized_prices.multiply(allocs_optimized, axis=1)
    port_val = prices_alloc.sum(axis=1)

    # COMPUTING STATS
    cr = (port_val.iloc[-1] - port_val.iloc[0]) / port_val.iloc[0]
    daily_return = port_val.pct_change()
    adr = daily_return.mean()
    sddr = daily_return.std()
    rfr = 0
    sr = (adr - rfr) * np.sqrt(len(normalized_prices))/sddr

    # PLOTTING THE PORTFOLIO VS PRICES_SPY
    if gen_plot:
        prices_SPY = prices_SPY/prices_SPY.iloc[0]
        df_temp = pd.concat([port_val, prices_SPY], keys=[
                            'Portfolio', 'SPY'], axis=1)
        plt.figure(1)
        df_temp.plot()
        plt.title("Daily Portfolio Value Post-Optimization vs SPY")
        plt.xlabel('DateTime')
        plt.ylabel('Normalized Price Value')
        plt.legend()
        plt.grid()
        plt.savefig("Figure1.png")

    allocs = allocs_optimized

    return allocs, cr, adr, sddr, sr


def get_sharpe_ratio(allocs, normalized_prices, rfr=0,):
    prices_alloc = normalized_prices.multiply(allocs, axis=1)
    port_val = prices_alloc.sum(axis=1)
    cr = (port_val.iloc[-1] - port_val.iloc[0]) / port_val.iloc[0]
    daily_return = port_val.pct_change()
    adr = daily_return.mean()
    sddr = daily_return.std()

    sharpe_ratio = (adr - rfr) * np.sqrt(len(normalized_prices))/sddr
    return -1 * sharpe_ratio


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    # HW2 GRAPH
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date,
                                                        syms=symbols,
                                                        gen_plot=True)

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
