"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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

Student Name: Jae Ro
"""

import datetime as dt
import pandas as pd
import util as ut
import random
from marketsimcode import compute_portvals
from indicators import *
import matplotlib.pyplot as plt
import QLearner as ql


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = ql.QLearner(num_states=4000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)
        self.sma_bins = None
        self.bb_bins = None
        self.ichi_bins_above = None
        self.ichi_bins_below = None
        self.ichi_bins_within = None
        self.order_book = None

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="AAPL",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 12, 31),
                    sv=100000):

        # Compensating for rolling metrics that cover previous n amount of days (so that we can have metrics on the first trading day)
        offset_date = sd - dt.timedelta(days=104)

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(offset_date, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose:
            print(prices)

        # Calculate Indicators
        df_inds = calculate_indicators(prices, symbol)
        last_price_date = df_inds['price'].last_valid_index()

        # PRICE-to-SMA
        price_sma = df_inds['price_sma_ratio']

        # BOLLINGER BAND POSITION
        bb_pos = df_inds['bb_position']

        # Transform Continuous Indicator Values to Discrete State Values
        df_discrete_inds = pd.DataFrame(index=prices.index)
        df_discrete_inds['price_normalized'] = df_inds['price']
        df_discrete_inds['price_change'] = df_discrete_inds['price_normalized'].pct_change()
        df_discrete_inds['price_sma_state'] = self.makeDiscreteSMA(price_sma)
        df_discrete_inds['bb_state'] = self.makeDiscreteBB(bb_pos)
        df_discrete_inds['macd_state'] = self.macdDiscrete(df_inds, prices)
        df_discrete_inds['ichi_state'] = self.ichimokuDiscrete(df_inds, prices)
        df_discrete_inds['state_value'] = df_discrete_inds['macd_state'].astype(str, skipna=True) + df_discrete_inds['price_sma_state'].astype(
            str, skipna=True) + df_discrete_inds['bb_state'].astype(str, skipna=True) + df_discrete_inds['ichi_state'].astype(str, skipna=True)

        # Finalizing our Indicators Dataframe and prepping for QLearning
        df_qtrade = df_discrete_inds[sd:]

        # Defining the initial state for Q-table
        start_state = df_qtrade.iloc[0]['state_value']
        start_state = int(start_state)

        # Creating the Order Book
        df_orders = pd.DataFrame(index=df_qtrade.index)
        df_orders['Symbol'] = symbol
        df_orders['Shares'] = 0
        df_orders['Order'] = 'HOLD'
        df_orders['Holdings'] = 0
        df_orders_prev = df_orders.copy()

        # Initialize the Q-Learner start state
        self.learner.querysetstate(start_state)

        # Start Training
        i = 0
        convergence_count = 0

        while i < 200:
            # Initialize Reward and Holdings
            reward = 0
            holdings = 0

            # Base Case Test for Conversion
            if (df_orders.equals(df_orders_prev) and i > 20):
                convergence_count += 1
                if (convergence_count >= 10):
                    # print("BREAKS AT i= ", i)
                    break
            else:
                convergence_count = 0

            # Storing current orders as Previous orders
            df_orders_prev = df_orders.copy()

            # Creating a new df_orders dataframe
            df_orders = pd.DataFrame(index=df_qtrade.index)
            df_orders['Symbol'] = symbol
            df_orders['Shares'] = 0
            df_orders['Order'] = 'HOLD'

            for day, value in df_qtrade.iterrows():
                reward = (1-self.impact) * holdings * df_qtrade.loc[day]['price_change']

                action = self.learner.query(int(df_qtrade.loc[day]['state_value']), reward)

                if action == 1 and (holdings == 0 or holdings == -1000):
                    df_orders.at[day, 'Order'] = 'BUY'
                    if holdings == -1000:
                        df_orders.at[day, 'Shares'] = 2000
                        holdings += 2000
                        # df_orders.at[day, 'Holdings'] = holdings
                    else:
                        df_orders.at[day, 'Shares'] = 1000
                        holdings += 1000
                        # df_orders.at[day, 'Holdings'] = holdings
                elif action == 2 and (holdings == 0 or holdings == 1000):
                    df_orders.at[day, 'Order'] = 'SELL'
                    if holdings == 1000:
                        df_orders.at[day, 'Shares'] = 2000
                        holdings -= 2000
                        # df_orders.at[day, 'Holdings'] = holdings
                    else:
                        df_orders.at[day, 'Shares'] = 1000
                        holdings -= 1000
                        # df_orders.at[day, 'Holdings'] = holdings

            i += 1

        # # Calculating Portvals
        # if last_price_date != df_orders.last_valid_index() or df_orders.at[last_price_date, 'Order'] == 'HOLD':
        #     df_orders = df_orders.append(pd.DataFrame({'Symbol': symbol, 'Shares': 0, 'Order': None}, index=[last_price_date]))

        self.setOrderBook(df_orders)
        # print("--Training--")
        # print(df_orders)

        # temp = df_orders[df_orders['Order'] != 'HOLD']
        # df_orders = pd.DataFrame(temp)
        # df_orders.reset_index(inplace=True)
        # df_orders.rename(columns={'index': 'Date'}, inplace=True)
        # portvals = compute_portvals(df_orders, sv, 0, self.impact)
        # # print("---- TRAINING SET ----")
        # # print(portvals.tail(60))

    def testPolicy(self, symbol="AAPL",
                   sd=dt.datetime(2010, 1, 1),
                   ed=dt.datetime(2011, 12, 31),
                   sv=100000):

        # Compensating for rolling metrics that cover previous n amount of days (so that we can have metrics on the first trading day)
        offset_date = sd - dt.timedelta(days=104)

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(offset_date, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose:
            print(prices)

        # Calculate Indicators
        df_inds = calculate_indicators(prices, symbol)

        last_price_date = df_inds['price'].last_valid_index()

        # PRICE-to-SMA
        price_sma = df_inds['price_sma_ratio']

        # BOLLINGER BAND POSITION
        bb_pos = df_inds['bb_position']

        # Transform Continuous Indicator Values to Discrete State Values
        df_discrete_inds = pd.DataFrame(index=prices.index)
        df_discrete_inds['price_normalized'] = df_inds['price']
        df_discrete_inds['price_change'] = df_discrete_inds['price_normalized'].pct_change()
        df_discrete_inds['price_sma_state'] = self.makeDiscreteSMATest(price_sma, bins=self.sma_bins)
        df_discrete_inds['bb_state'] = self.makeDiscreteBBTest(bb_pos, bins=self.bb_bins)
        df_discrete_inds['macd_state'] = self.macdDiscrete(df_inds, prices)
        df_discrete_inds['ichi_state'] = self.ichimokuDiscreteTest(df_inds, prices, self.ichi_bins_above, self.ichi_bins_below, self.ichi_bins_within)
        df_discrete_inds['state_value'] = df_discrete_inds['macd_state'].astype(str, skipna=True) + df_discrete_inds['price_sma_state'].astype(
            str, skipna=True) + df_discrete_inds['bb_state'].astype(str, skipna=True) + df_discrete_inds['ichi_state'].astype(str, skipna=True)

        # Finalizing our Indicators Dataframe and prepping for QLearning
        df_qtrade = df_discrete_inds[sd:]
        df_qtrade.dropna(inplace=True)

        # Defining the initial state for Q-table
        start_state = df_qtrade.iloc[0]['state_value']
        start_state = int(start_state)

        # Creating the Order Book
        df_orders = pd.DataFrame(index=df_qtrade.index)
        df_orders['Symbol'] = symbol
        df_orders['Shares'] = 0
        df_orders['Order'] = 'HOLD'

        # Initialize the Q-Learner start state
        self.learner.querysetstate(start_state)

        # Initialize Reward and Holdings
        holdings = 0

        for day, value in df_qtrade.iterrows():
            action = self.learner.querysetstate(int(df_qtrade.loc[day]['state_value']))

            if action == 1 and (holdings == 0 or holdings == -1000):
                df_orders.at[day, 'Order'] = 'BUY'
                if holdings == -1000:
                    df_orders.at[day, 'Shares'] = 2000
                    holdings += 2000

                else:
                    df_orders.at[day, 'Shares'] = 1000
                    holdings += 1000

            elif action == 2 and (holdings == 0 or holdings == 1000):
                df_orders.at[day, 'Order'] = 'SELL'
                if holdings == 1000:
                    df_orders.at[day, 'Shares'] = 2000
                    holdings -= 2000

                else:
                    df_orders.at[day, 'Shares'] = 1000
                    holdings -= 1000

        # if last_price_date != df_orders.last_valid_index() or df_orders.at[last_price_date, 'Order'] == 'HOLD':
        #     df_orders = df_orders.append(pd.DataFrame({'Symbol': symbol, 'Shares': 0, 'Order': None}, index=[last_price_date]))

        self.setOrderBook(df_orders)

        # # Calculate Portvals
        # df_orders_port = df_orders.copy(deep=True)
        # temp = df_orders_port[df_orders_port['Order'] != 'HOLD']
        # df_orders_port = pd.DataFrame(temp)
        # df_orders_port.reset_index(inplace=True)
        # df_orders_port.rename(columns={'index': 'Date'}, inplace=True)
        # portvals = compute_portvals(df_orders_port, sv, 0, self.impact)
        # # print("---- TESTING SET -----")
        # # print(portvals.tail(60))

        # Return Order Book (single column denoting shares with date index)
        df_orders.loc[df_orders['Order'] == 'SELL', 'Shares'] *= -1
        df_orders = df_orders[['Shares']]
        df_orders.rename(columns={"Shares": symbol}, inplace=True)

        # print("--Testing--")
        # print(df_orders)
        return df_orders

    def setOrderBook(self, df_orders):
        self.order_book = df_orders

    def getOrderBook(self):
        return self.order_book

    def makeDiscreteSMA(self, indicator):
        min_val = indicator.min() - 0.000001
        max_val = indicator.max() + 0.000001
        bins = np.linspace(min_val, max_val, 11)
        self.sma_bins = bins

        group_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        indicator_states = pd.cut(indicator, bins, labels=group_numbers)

        return indicator_states

    def makeDiscreteBB(self, indicator):
        min_val = indicator.min() - 0.000001
        max_val = indicator.max() + 0.000001
        bins = np.linspace(min_val, max_val, 11)
        self.bb_bins = bins

        group_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        indicator_states = pd.cut(indicator, bins, labels=group_numbers)
        return indicator_states

    def macdDiscrete(self, indicators, prices):
        macd = indicators['macd_value']
        signal = indicators['signal']
        divergence = indicators['divergence']
        df_mac = pd.DataFrame(index=prices.index)
        df_mac['macd_choice'] = np.NaN
        macd_choice = df_mac['macd_choice']
        past_divergence = divergence.shift(1)

        macd_choice[divergence.lt(0) & past_divergence.lt(0)] = '0'
        macd_choice[divergence.lt(0) & past_divergence.gt(0)] = '1'
        macd_choice[divergence.gt(0) & past_divergence.lt(0)] = '2'
        macd_choice[divergence.gt(0) & past_divergence.gt(0)] = '3'

        # print(macd_choice.describe())

        return macd_choice

    def ichimokuDiscrete(self, indicators, prices):

        leadspan_a = indicators['leadspan_a']
        leadspan_b = indicators['leadspan_b']
        share_price = indicators['price']

        df_ichimoku = pd.DataFrame(index=prices.index)
        df_ichimoku['price'] = share_price
        df_ichimoku['lead_a'] = leadspan_a
        df_ichimoku['lead_b'] = leadspan_b

        df_ichimoku['cloud_choice'] = np.NaN
        cloud_choice = df_ichimoku['cloud_choice']

        cloud_choice[share_price.gt(leadspan_a) & share_price.gt(leadspan_b)] = "above"  # above the cloud
        cloud_choice[share_price.lt(leadspan_a) & share_price.lt(leadspan_b)] = "below"  # below the cloud
        cloud_choice[share_price.gt(leadspan_a) & share_price.lt(leadspan_b)] = "within"  # within the cloud
        cloud_choice[share_price.lt(leadspan_a) & share_price.gt(leadspan_b)] = "within"  # within the cloud

        cloud_above = df_ichimoku[cloud_choice.eq("above")]
        cloud_below = df_ichimoku[cloud_choice.eq("below")]
        cloud_within = df_ichimoku[cloud_choice.eq("within")]

        # Binning the Above Cloud Data
        cloud_above['magnitude_a'] = abs(leadspan_a - cloud_above['price'])
        cloud_above['magnitude_b'] = abs(leadspan_b - cloud_above['price'])
        cloud_above['magnitude'] = cloud_above[['magnitude_a', 'magnitude_b']].min(axis=1)

        above_max = cloud_above['magnitude'].max() + 0.000001

        above_min = cloud_above['magnitude'].min() - 0.000001

        above_bins = np.linspace(above_min, above_max, 5)
        self.ichi_bins_above = above_bins
        above_group_numbers = ["6", "7", "8", "9"]
        above_cloud_states = pd.cut(cloud_above['magnitude'], above_bins, labels=above_group_numbers)

        # Binning the Below Cloud Data
        cloud_below['magnitude_a'] = abs(leadspan_a - cloud_below['price'])
        cloud_below['magnitude_b'] = abs(leadspan_b - cloud_below['price'])
        cloud_below['magnitude'] = cloud_below[['magnitude_a', 'magnitude_b']].min(axis=1)

        below_max = cloud_below['magnitude'].max() + 0.000001
        below_min = cloud_below['magnitude'].min() - 0.000001
        below_bins = np.linspace(below_min, below_max, 5)
        self.ichi_bins_below = below_bins
        below_group_numbers = ["0", "1", "2", "3"]
        below_cloud_states = pd.cut(cloud_below['magnitude'], below_bins, labels=below_group_numbers)

        magnitudes = cloud_above['magnitude'].append(cloud_below['magnitude'])
        df_ichimoku['magnitude'] = magnitudes.sort_index(axis=0)

        # Binning the Within Cloud Data
        within_max = cloud_within['price'].max() + 1
        within_min = cloud_within['price'].min() - 1
        within_bins = np.linspace(within_min, within_max, 3)
        self.ichi_bins_within = within_bins

        within_group_numbers = ["4", "5"]
        within_cloud_states = pd.cut(cloud_within['price'], within_bins, labels=within_group_numbers)

        # Appending all above, below, and within state labels to single column
        above_below = above_cloud_states.append(below_cloud_states)
        above_within_below = above_below.append(within_cloud_states)
        above_within_below = above_within_below.sort_index(axis=0)
        df_ichimoku['state_value'] = np.NaN
        df_ichimoku['state_value'] = above_within_below

        return df_ichimoku['state_value']

    def makeDiscreteSMATest(self, indicator, bins):

        group_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        indicator_states = pd.cut(indicator, bins, labels=group_numbers)

        return indicator_states

    def makeDiscreteBBTest(self, indicator, bins):

        group_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        indicator_states = pd.cut(indicator, bins, labels=group_numbers)

        return indicator_states

    def ichimokuDiscreteTest(self, indicators, prices, above_bins, below_bins, within_bins):

        leadspan_a = indicators['leadspan_a']
        leadspan_b = indicators['leadspan_b']
        share_price = indicators['price']

        df_ichimoku = pd.DataFrame(index=prices.index)
        df_ichimoku['price'] = share_price
        df_ichimoku['lead_a'] = leadspan_a
        df_ichimoku['lead_b'] = leadspan_b

        df_ichimoku['cloud_choice'] = np.NaN
        cloud_choice = df_ichimoku['cloud_choice']

        cloud_choice[share_price.gt(leadspan_a) & share_price.gt(leadspan_b)] = "above"  # above the cloud
        cloud_choice[share_price.lt(leadspan_a) & share_price.lt(leadspan_b)] = "below"  # below the cloud
        cloud_choice[share_price.gt(leadspan_a) & share_price.lt(leadspan_b)] = "within"  # within the cloud
        cloud_choice[share_price.lt(leadspan_a) & share_price.gt(leadspan_b)] = "within"  # within the cloud

        cloud_above = df_ichimoku[cloud_choice.eq("above")]
        cloud_below = df_ichimoku[cloud_choice.eq("below")]
        cloud_within = df_ichimoku[cloud_choice.eq("within")]

        # Binning the Above Cloud Data
        cloud_above['magnitude_a'] = abs(leadspan_a - cloud_above['price'])
        cloud_above['magnitude_b'] = abs(leadspan_b - cloud_above['price'])
        cloud_above['magnitude'] = cloud_above[['magnitude_a', 'magnitude_b']].min(axis=1)

        above_group_numbers = ["6", "7", "8", "9"]
        above_cloud_states = pd.cut(cloud_above['magnitude'], above_bins, labels=above_group_numbers)

        # Binning the Below Cloud Data
        cloud_below['magnitude_a'] = abs(leadspan_a - cloud_below['price'])
        cloud_below['magnitude_b'] = abs(leadspan_b - cloud_below['price'])
        cloud_below['magnitude'] = cloud_below[['magnitude_a', 'magnitude_b']].min(axis=1)

        below_group_numbers = ["0", "1", "2", "3"]
        below_cloud_states = pd.cut(cloud_below['magnitude'], below_bins, labels=below_group_numbers)

        magnitudes = cloud_above['magnitude'].append(cloud_below['magnitude'])
        df_ichimoku['magnitude'] = magnitudes.sort_index(axis=0)

        # Binning the Within Cloud Data
        within_group_numbers = ["4", "5"]
        within_cloud_states = pd.cut(cloud_within['price'], within_bins, labels=within_group_numbers)

        # Appending all above, below, and within state labels to single column
        above_below = above_cloud_states.append(below_cloud_states)
        above_within_below = above_below.append(within_cloud_states)
        above_within_below = above_within_below.sort_index(axis=0)
        df_ichimoku['state_value'] = np.NaN
        df_ichimoku['state_value'] = above_within_below

        return df_ichimoku['state_value']

    # this method should use the existing policy and test it against new data


if __name__ == "__main__":
    strat = StrategyLearner()
    strat.addEvidence()
    strat.testPolicy()
    print("One does not simply think up a strategy")


def author():
    return 'jro32'
