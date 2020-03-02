import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals


def test_policy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)[symbol]
    normalized_prices = prices/prices[0]
    df_prices = pd.DataFrame(normalized_prices, index=normalized_prices.index)

    df_orders = pd.DataFrame(index=df_prices.index)
    df_orders['Symbol'] = symbol
    df_orders['Shares'] = 1000
    df_orders['Order'] = 'HOLD'
    df_future = df_prices.shift(-1)
    df_orders['Order'] = df_prices < df_future
    df_orders['Order'] = np.where(df_orders['Order'] == True, 'BUY', 'SELL')

    # Previous Days Orders
    orders_past = pd.DataFrame(df_orders)
    orders_past = orders_past.shift(1)
    orders_past['Order'] = np.where(orders_past['Order'] == 'BUY', 'SELL', 'BUY')

    # Next Days Orders
    df_orders = df_orders.append(orders_past)
    df_orders = df_orders.dropna()
    df_orders = df_orders.reset_index()
    df_orders.rename(columns={'index': 'Date'}, inplace=True)

    return df_orders


def plot_results():
    train_sd = dt.datetime(2008, 1, 1)
    train_ed = dt.datetime(2009, 12, 31)
    test_sd = dt.datetime(2010, 1, 1)
    test_ed = dt.datetime(2011, 12, 31)
    dates = [(train_sd, train_ed), (test_sd, test_ed)]

    start_val = 100000
    commission = 0
    impact = 0

    for i, date in enumerate(dates):
        title = ''
        if i == 0:
            title = 'IN SAMPLE'
        else:
            title = 'OUT OF SAMPLE'

        # OPTIMAL STRATEGY
        start = date[0]
        end = date[1]
        strategy = test_policy(sd=start, ed=end, sv=start_val)
        strategy_portvals = compute_portvals(strategy, start_val, commission, impact)

        # BENCHMARK STRATEGY
        prices = get_data(['JPM'], pd.date_range(start, end))['JPM']
        benchmark_orders = pd.DataFrame({'Symbol': 'JPM', 'Shares': 0, 'Order': None}, index=prices.index)
        first = benchmark_orders.first_valid_index()
        benchmark_orders.at[first, 'Order'] = 'BUY'
        benchmark_orders.at[first, 'Shares'] = 1000
        benchmark_orders.reset_index(inplace=True)
        benchmark_orders.rename(columns={'index': 'Date'}, inplace=True)
        benchmark_portvals = compute_portvals(benchmark_orders, start_val, commission, impact)

        calc_return(strategy_portvals, start, end, "Theoretically Optimal", title)

        # Plotting Optimal vs Benchmark
        plt.figure()
        normalized_strat = strategy_portvals/start_val
        normalized_bench = benchmark_portvals/start_val
        normalized_strat.plot(color='red')
        normalized_bench.plot(color='green')
        plt.legend(['Theoretically Optimal Strategy', 'Benchmark Strategy'])
        plt.xlabel('Datetime')
        plt.ylabel('USD')
        plt.title('Theoretically Optimal Strategy vs Benchmark Strategy %s' % title)
        plt.savefig('%s_theoretically_optimal.png' % title)
        plt.grid(color='grey', linestyle='-', linewidth=0.25)
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


if __name__ == "__main__":
    plot_results()
