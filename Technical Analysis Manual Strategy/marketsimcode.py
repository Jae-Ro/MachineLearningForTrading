"""MC2-P1: Market simulator.

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

Student Name: Tucker Balch (replace with your name)

"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt


def compute_portvals(orders_df, start_val=100000, commission=9.95, impact=0.005):
    # print(orders_df)
    # orders_df.reset_index(inplace=True)

    orders_df['Date'] = pd.to_datetime(orders_df['Date'])
    df_orders = orders_df
    # print(df_orders)

    # Locating SELL orders in the orders dataframe and multiplying by -1
    df_orders.loc[df_orders['Order'] == 'SELL', 'Shares'] *= -1

    # Sorting the orders by date
    df_orders = df_orders.sort_values(by=['Date'])

    # Checking for missing values in the order book
    nan_count = df_orders.isnull().sum().sum()
    # if nan_count > 0:
    #     df_orders = df_orders.ffil(axis=0)
    #     df_orders = df_orders.ffil(axis=0)

    # Pivot table method (without using for loop)
    table = pd.pivot_table(df_orders, values='Shares', index=[
        'Date'], columns=['Symbol'], fill_value=0, aggfunc=np.sum)

    # Counting number of transactions per date
    transaction_count = df_orders.set_index(
        ['Date', 'Symbol', 'Shares']).count(level=0)

    # Calculating Impact by accounting for BUY and SELL of same stock in given day (make all values positive so no canceling out)
    df_orders['Shares'] = abs(df_orders['Shares'])
    penalty_count = pd.pivot_table(df_orders, values='Shares', index=[
        'Date'], columns=['Symbol'], fill_value=0, aggfunc=np.sum)

    transaction_count['Count'] = transaction_count
    transaction_count = transaction_count.loc[:, ['Count']]

    transaction_fee = transaction_count * 0 + commission * -1
    transaction_fee['Fee'] = transaction_fee
    transaction_fee = transaction_fee.loc[:, ['Fee']]

    # FIND START AND END DATE OF ORDERS
    start_date = df_orders['Date'].iloc[0]
    end_date = df_orders['Date'].iloc[-1]
    dates = pd.date_range(start_date, end_date)

    # FIND ALL UNIQUE STOCKS BEING ASSESSED
    stock_symbols = list(set(df_orders['Symbol']))

    # GET SPECIFIC STOCK DATA
    port_vals = get_data(stock_symbols, dates)
    port_vals = port_vals[stock_symbols]

    # Calculating Number of Shares per stock
    df_trades = port_vals.copy()
    df_trades[stock_symbols] = 0

    filled_trades = df_trades + table
    filled_trades = filled_trades.fillna(0)
    filled_trades = filled_trades.astype('int64')
    df_stockholdings = filled_trades.cumsum()

    # Calculating Cash Data
    df_marketval = filled_trades * port_vals * -1

    df_positions = df_marketval * -1
    df_positions['StockSum'] = df_positions.sum(axis=1)
    df_stockvalues = df_stockholdings * port_vals
    df_stockvalues = df_stockvalues.sum(axis=1)

    # COMMISSION and IMPACT
    penalty_trades = df_trades + penalty_count
    penalty_trades = penalty_trades.fillna(0)
    penalty_trades = penalty_trades * port_vals
    df_penalty = abs(penalty_trades * impact) * -1

    df_penalty = df_penalty.sum(axis=1)
    df_penalty = df_penalty.to_frame()
    df_penalty['Penalty'] = df_penalty
    df_penalty = df_penalty.loc[:, ['Penalty']]
    df_cost = pd.concat([df_penalty, transaction_count, transaction_fee], axis=1).fillna(0)

    df_cost['Cost'] = df_cost['Penalty'] + (df_cost['Count'] * commission * -1)
    df_cost = df_cost.loc[:, ['Cost']]

    df_positions = pd.concat([df_positions, df_cost], axis=1).fillna(0)
    df_cash = df_marketval
    df_positions['DeltaCash'] = df_cash.sum(axis=1) + df_positions['Cost']
    df_positions['Value'] = 0
    df_positions['StockCash'] = 0
    df_positions['StockCash'] = df_positions['StockCash'] + df_stockvalues

    df_positions.iloc[:1, -2] = df_positions.iloc[:1, -3] + start_val
    df_positions.iloc[1:, -2] = df_positions.iloc[1:, -3] + df_positions.iloc[1:, -2]

    df_positions['Cash'] = df_positions['Value'].cumsum()
    df_positions['PortVal'] = df_positions['Cash'] + df_positions['StockCash']

    port_vals = df_positions['PortVal']

    # print(df_positions)
    return port_vals


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-12.csv"
    sv = 100000
    com = 0
    imp = 0.005
    df_orders = pd.read_csv(of)

    # Process orders
    portvals = compute_portvals(
        orders_df=df_orders, start_val=sv, commission=com, impact=imp)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)

    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2, 0.01, 0.02, 1.5]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


def author():
    return 'jro32'


if __name__ == "__main__":
    test_code()
