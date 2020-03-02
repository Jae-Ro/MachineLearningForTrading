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
from decimal import Decimal


class experiment2 (object):

    def __init__(self):
        self.strategy_learner = None
        self.impact_vals = np.linspace(0, 0.9, 10)
        # self.impact_vals = [0, 0.0026, 0.0053, 0.0079, 0.0105, 0.0132, 0.0158, 0.0184, 0.0211, 0.0237,
        #                     0.0263, 0.0289, 0.0316, 0.0342, 0.0368, 0.0395, 0.0421, 0.0447, 0.0474, 0.05, 0.1, 0.5]
        self.columns = ["cr", "num_orders"]

        self.df_manual = pd.DataFrame(index=self.impact_vals, columns=self.columns)
        self.df_bench = pd.DataFrame(index=self.impact_vals,  columns=self.columns)
        self.df_learner = pd.DataFrame(index=self.impact_vals, columns=self.columns)

        self.df_results = pd.DataFrame(index=self.impact_vals)

    def plot_results(self):
        train_sd = dt.datetime(2008, 1, 1)
        train_ed = dt.datetime(2009, 12, 31)
        test_sd = dt.datetime(2010, 1, 1)
        test_ed = dt.datetime(2011, 12, 31)
        dates = [(train_sd, train_ed), (test_sd, test_ed)]

        start_val = 100000
        commission = 0
        impact_vals = self.impact_vals

        for i, imp in enumerate(impact_vals):
            title = 'In-SAMPLE (impact value: %s)' % impact_vals[i]
            impact = impact_vals[i]
            start = dates[0][0]
            end = dates[0][1]

            # MANUAL STRATEGY
            manual_strategy = ms.test_policy(sd=start, ed=end, sv=start_val)
            manual_strategy_portvals = compute_portvals(manual_strategy, start_val, commission, impact)

            # STRATEGY LEARNER
            df_orders = None
            self.strategy_learner = sl.StrategyLearner(False, 0.0)
            self.strategy_learner.addEvidence(symbol='JPM')
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

            benchmark_portvals = compute_portvals(benchmark_orders, start_val, commission, impact)

            # CALCULATING CUMULATIVE RETURNS OF Manual, Benchmark, Learner
            manual_rets = self.calc_return(manual_strategy_portvals, start, end, "Manual Strategy", title)
            bench_rets = self.calc_return(benchmark_portvals, start, end, "Benchmark Strategy", title)
            learner_rets = self.calc_return(strat_learner_portvals, start, end, "Strategy Learner", title)

            self.df_manual.at[impact, "cr"] = manual_rets
            self.df_bench.at[impact, "cr"] = bench_rets
            self.df_learner.at[impact, "cr"] = learner_rets

            # Calculate Number of Orders
            manual_orders = len(manual_strategy.index)
            bench_orders = len(benchmark_orders.index)
            learner_orders = len(df_orders.index)

            self.df_manual.at[impact, "num_orders"] = manual_orders
            self.df_bench.at[impact, "num_orders"] = bench_orders
            self.df_learner.at[impact, "num_orders"] = learner_orders

        self.df_manual.to_csv('experiment2_manual.csv')
        self.df_bench.to_csv('experiment2_bench.csv')
        self.df_learner.to_csv('experiment2_learner.csv')

        # Cumulative Return
        plt.figure()
        plt.title('Impact vs Cumulative Return')
        plt.plot(self.df_manual[['cr']], color='r')
        plt.plot(self.df_bench[['cr']], color='g')
        plt.plot(self.df_learner[['cr']], color='b')
        plt.xlabel('Impact Level')
        plt.ylabel('USD')
        plt.grid(color='grey', linestyle='-', linewidth=0.25)
        plt.legend(['Manual Strategy', 'Benchmark Strategy', 'Strategy Learner'])
        plt.savefig('experiment_2_impact_cr')
        # plt.show()

        # Number Orders
        plt.figure()
        plt.title('Impact vs Number of Orders')
        plt.plot(self.df_manual[['num_orders']], color='r')
        plt.plot(self.df_bench[['num_orders']], color='g')
        plt.plot(self.df_learner[['num_orders']], color='b')
        plt.xlabel('Impact Level')
        plt.ylabel('Number of Orders')
        plt.grid(color='grey', linestyle='-', linewidth=0.25)
        plt.legend(['Manual Strategy', 'Benchmark Strategy', 'Strategy Learner'])
        plt.savefig('experiment_2_impact_num_orders')
        # plt.show()

    def calc_return(self, port_val, start_date, end_date, title, set):
        cr = (port_val.iloc[-1] - port_val.iloc[0]) / port_val.iloc[0]
        daily_return = port_val.pct_change()
        adr = daily_return.mean()
        sddr = daily_return.std()

        # print("\n")
        # print(f"Statistics of {title} Portfolio {set}")
        # print(f"Start Date: {start_date}")
        # print(f"End Date: {end_date}")
        # print(f"Volatility (stdev of daily returns): {sddr}")
        # print(f"Average Daily Return: {adr}")
        # print(f"Cumulative Return: {cr}")

        return cr


def author():
    return 'jro32'


if __name__ == "__main__":
    exp = experiment2()
    exp.plot_results()
