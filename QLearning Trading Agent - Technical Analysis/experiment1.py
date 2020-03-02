import datetime as dt
import pandas as pd
import util as ut
import random
from marketsimcode import compute_portvals
from indicators import *
import matplotlib.pyplot as plt
import QLearner as ql
import StrategyLearner as sl
import ManualStrategy as ms


class experiment1 (object):

    def __init__(self):
        self.strategy_learner = None

    def plot_results(self):
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

            start = date[0]
            end = date[1]

            # MANUAL STRATEGY
            manual_strategy = ms.test_policy(sd=start, ed=end, sv=start_val)
            manual_strategy_portvals = compute_portvals(manual_strategy, start_val, commission, impact)

            # STRATEGY LEARNER
            df_orders = None
            if i == 0:
                self.strategy_learner = sl.StrategyLearner(False, 0.0)
                self.strategy_learner.addEvidence(symbol='JPM')
                df_orders = self.strategy_learner.getOrderBook()
            else:
                self.strategy_learner.testPolicy(symbol='JPM', sd=start, ed=end, sv=start_val)
                df_orders = self.strategy_learner.getOrderBook()

            # print(df_orders)
            temp_orders = df_orders[df_orders['Order'] != 'HOLD']
            df_orders = pd.DataFrame(temp_orders)
            df_orders.reset_index(inplace=True)
            df_orders.rename(columns={'index': 'Date'}, inplace=True)
            strat_learner_portvals = compute_portvals(df_orders, start_val, commission, impact)
            # print(strat_learner_portvals.tail(60))

            # BENCHMARK
            prices = ut.get_data(['JPM'], pd.date_range(start, end))['JPM']
            benchmark_orders = pd.DataFrame({'Symbol': 'JPM', 'Shares': 0, 'Order': None}, index=prices.index)
            first = benchmark_orders.first_valid_index()
            last = benchmark_orders.last_valid_index()
            benchmark_orders.at[first, 'Order'] = 'BUY'
            benchmark_orders.at[first, 'Shares'] = 1000
            benchmark_orders.at[last, 'Order'] = 'HOLD'
            benchmark_orders.at[last, 'Shares'] = 0
            benchmark_orders = benchmark_orders.dropna()
            benchmark_orders.at[last, 'Order'] = 'SELL'
            benchmark_orders.at[last, 'Shares'] = 1000

            last_price_date = prices.last_valid_index()
            if last_price_date != benchmark_orders.last_valid_index():
                benchmark_orders = benchmark_orders.append(pd.DataFrame({'Symbol': symbol, 'Shares': 0, 'Order': None}, index=[last_price_date]))

            benchmark_orders.reset_index(inplace=True)
            benchmark_orders.rename(columns={'index': 'Date'}, inplace=True)
            # print('--BENCHMARK ORDERS---')
            # print(benchmark_orders)
            benchmark_portvals = compute_portvals(benchmark_orders, start_val, commission, impact)
            # print("---BENCHMARK PORT VALS---")
            # print(benchmark_portvals)

            # CALCULATING CUMULATIVE RETURNS OF Manual, Benchmark, Learner
            self.calc_return(manual_strategy_portvals, start, end, "Manual Strategy", title)
            self.calc_return(benchmark_portvals, start, end, "Benchmark Strategy", title)
            self.calc_return(strat_learner_portvals, start, end, "Strategy Learner", title)

            # Plotting Manual vs Benchmark vs Strategy Learner
            plt.figure()
            normalized_strat = manual_strategy_portvals/start_val
            normalized_bench = benchmark_portvals/start_val
            normalalized_learner = strat_learner_portvals/start_val

            plt.plot(normalized_strat, color='r')
            plt.plot(normalized_bench, color='g')
            plt.plot(normalalized_learner, color='b')
            plt.xlabel('Datetime')
            plt.ylabel('USD')
            plt.title('Manual Strategy vs Benchmark Strategy vs Strategy Learner %s' % title)

            plt.grid(color='grey', linestyle='-', linewidth=0.25)

            plt.legend(['Manual Strategy', 'Benchmark Strategy', 'Strategy Learner', 'Going Long', 'Going Short'])
            plt.savefig('%s_experiment_1_strategy_comparison.png' % title)
            # plt.show()

    def calc_return(self, port_val, start_date, end_date, title, set):
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
    exp = experiment1()
    exp.plot_results()
