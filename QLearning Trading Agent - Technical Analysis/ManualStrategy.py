import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
from indicators import *


def test_policy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):

    date_range = pd.date_range(sd, ed)
    prices = get_data([symbol], date_range)
    df_inds = calculate_indicators(prices, symbol)

    original_price = df_inds['price']
    last_price_date = original_price.last_valid_index()

    # PRICE-to-SMA
    price_sma = df_inds['price_sma_ratio']

    # BOLLINGER BAND POSITION
    bb_pos = df_inds['bb_position']

    # MACD DIVERGENCE
    macd = df_inds['macd_value']
    signal = df_inds['signal']
    divergence = df_inds['divergence']

    # ICHIMOKU KINKO KYO CLOUD
    leadspan_a = df_inds['leadspan_a']
    leadspan_b = df_inds['leadspan_b']

    # FORMING THE STRATEGY: 0 = hold, 1 = go long, -1 = go short
    strategy = pd.DataFrame(index=prices.index)
    strategy['sma_choice'] = 0
    strategy['bb_choice'] = 0
    strategy['macd_choice'] = 0
    strategy['cloud_choice'] = 0

    sma_choice = strategy['sma_choice']
    sma_choice[price_sma < 0.8] = 1  # price / sma is less than 99% (price below mean)
    sma_choice[price_sma > 1.2] = -1  # price / sma is greater than 101% (price above mean)

    bb_choice = strategy['bb_choice']
    bb_choice[bb_pos < -0.9] = 1  # less than 0.92 std deviations above the mean
    bb_choice[bb_pos > 0.9] = -1  # greater than 0.92 std deviations above the mean

    macd_choice = strategy['macd_choice']
    past_divergence = divergence.shift(1)
    macd_choice[divergence.gt(0) & past_divergence.lt(0)] = 1  # upward trend
    macd_choice[divergence.lt(0) & past_divergence.gt(0)] = -1  # downward trend

    cloud_choice = strategy['cloud_choice']
    leadingspans = [leadspan_a, leadspan_b]
    cloud_choice[original_price.lt(leadingspans[0]) & original_price.lt(leadingspans[1])] = 1  # below the cloud
    cloud_choice[original_price.gt(leadingspans[0]) & original_price.gt(leadingspans[1])] = -1  # above the cloud

    position = 0
    df_orders = pd.DataFrame(index=prices.index)
    df_orders['Symbol'] = symbol
    df_orders['Shares'] = 2000
    df_orders['Order'] = 'HOLD'

    for day, value in df_orders.iterrows():
        if (cloud_choice[day] != 0 or position != 0):
            if (((macd_choice[day] == 1) or (bb_choice[day] == 1 or sma_choice[day] == 1)) and position != 1):
                if (position != -1):
                    df_orders.at[day, 'Shares'] = 1000

                df_orders.at[day, 'Order'] = 'BUY'
                position = 1

            elif (((macd_choice[day] == -1) or (bb_choice[day] == -1 or sma_choice[day] == -1)) and position != -1):
                if position != 1:
                    df_orders.at[day, 'Shares'] = 1000

                df_orders.at[day, 'Order'] = 'SELL'
                position = -1

    if df_orders.iloc[0, 2] == 'HOLD':
        df_orders.iloc[0, 2] = None
        df_orders.iloc[0, 1] = 0

    temp = df_orders[df_orders['Order'] != 'HOLD']
    df_orders = pd.DataFrame(temp)

    if last_price_date != df_orders.last_valid_index():
        df_orders = df_orders.append(pd.DataFrame({'Symbol': symbol, 'Shares': 0, 'Order': None}, index=[last_price_date]))

    df_orders.reset_index(inplace=True)
    df_orders.rename(columns={'index': 'Date'}, inplace=True)

    return df_orders


def plot_results():
    train_sd = dt.datetime(2008, 1, 1)
    train_ed = dt.datetime(2009, 12, 31)
    test_sd = dt.datetime(2010, 1, 1)
    test_ed = dt.datetime(2011, 12, 31)
    dates = [(train_sd, train_ed), (test_sd, test_ed)]

    start_val = 100000
    commission = 9.95
    impact = 0.005

    for i, date in enumerate(dates):
        title = ''
        if i == 0:
            title = 'IN SAMPLE'
        else:
            title = 'OUT OF SAMPLE'

        # MANUAL STRATEGY
        start = date[0]
        end = date[1]
        strategy = test_policy(sd=start, ed=end, sv=start_val)
        strategy_portvals = compute_portvals(strategy, start_val, commission, impact)

        # BENCHMARK
        prices = get_data(['JPM'], pd.date_range(start, end))['JPM']
        benchmark_orders = pd.DataFrame({'Symbol': 'JPM', 'Shares': 0, 'Order': None}, index=prices.index)
        first = benchmark_orders.first_valid_index()
        last = benchmark_orders.last_valid_index()
        benchmark_orders.at[first, 'Order'] = 'BUY'
        benchmark_orders.at[first, 'Shares'] = 1000
        benchmark_orders.at[last, 'Order'] = 'HOLD'
        benchmark_orders.at[last, 'Shares'] = 0
        benchmark_orders = benchmark_orders.dropna()
        benchmark_orders.at[last, 'Order'] = None
        benchmark_orders.at[last, 'Shares'] = 0

        last_price_date = prices.last_valid_index()
        if last_price_date != benchmark_orders.last_valid_index():
            benchmark_orders = benchmark_orders.append(pd.DataFrame({'Symbol': symbol, 'Shares': 0, 'Order': None}, index=[last_price_date]))

        benchmark_orders.reset_index(inplace=True)
        benchmark_orders.rename(columns={'index': 'Date'}, inplace=True)

        benchmark_portvals = compute_portvals(benchmark_orders, start_val, commission, impact)

        calc_return(strategy_portvals, start, end, "Manual Strategy", title)
        calc_return(benchmark_portvals, start, end, "Benchmark Strategy", title)

        # Plotting Manual vs Benchmark
        plt.figure()
        normalized_strat = strategy_portvals/start_val
        normalized_bench = benchmark_portvals/start_val
        plt.plot(normalized_strat, color='r')
        plt.plot(normalized_bench, color='g')
        plt.xlabel('Datetime')
        plt.ylabel('USD')
        plt.title('Manual Strategy vs Benchmark Strategy %s' % title)

        plt.grid(color='grey', linestyle='-', linewidth=0.25)

        port = strategy.set_index('Date')

        for date, val in port.iterrows():
            if port.at[date, 'Order'] == 'BUY':
                plt.axvline(date, color='blue')
            elif port.at[date, 'Order'] == 'SELL':
                plt.axvline(date,  color='black')
        plt.legend(['Manual Strategy', 'Benchmark Strategy', 'Going Long', 'Going Short'])
        plt.savefig('%s_manual_strategy.png' % title)
        # plt.show()


def calc_return(port_val, start_date, end_date, title, set):
    cr = (port_val.iloc[-1] - port_val.iloc[0]) / port_val.iloc[0]
    daily_return = port_val.pct_change()
    adr = daily_return.mean()
    sddr = daily_return.std()

    print("\n")
    print(f"Statistics of {title} Portfolio {set}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


def author():
    return 'jro32'


def main():
    plot_results()


if __name__ == "__main__":
    main()
