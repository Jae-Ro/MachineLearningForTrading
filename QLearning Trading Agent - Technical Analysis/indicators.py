import pandas as pd
import numpy as np
import datetime as dt
import util as ut
import matplotlib.pyplot as plt


def calculate_indicators(df_prices, sym='JPM'):
    prices = df_prices[sym]
    prices = prices/prices[0]

    df_indicators = pd.DataFrame(index=prices.index)

    # 1) SIMPLE MOVING AVERAGE
    df_indicators['price'] = prices
    sma = prices.rolling(10).mean()
    df_indicators['sma'] = sma
    ratio = df_indicators['price']/df_indicators['sma']
    df_indicators['price_sma_ratio'] = ratio

    # 2) BOLLINGER BANDS
    stdev = df_indicators['price'].rolling(10).std()
    two_std = 2 * stdev
    upperband = df_indicators['sma'] + two_std
    lowerband = df_indicators['sma'] + two_std
    bbp = (df_indicators['price'] - df_indicators['sma']) / two_std
    df_indicators['upper_band'] = upperband
    df_indicators['lower_band'] = lowerband
    df_indicators['bb_position'] = bbp

    # 3) MACD

    twelve = df_indicators['price'].ewm(span=12).mean()
    twentysix = df_indicators['price'].ewm(span=26).mean()
    macd = twelve - twentysix
    signal = macd.ewm(span=9).mean()
    diverge = macd - signal

    df_indicators['ema_12'] = twelve
    df_indicators['ema_26'] = twentysix
    df_indicators['macd_value'] = macd
    df_indicators['signal'] = signal
    df_indicators['divergence'] = diverge

    # 4) ICHIMOKU KINKO HYO - Senkou Sen A (Leading Span A) and Senkou Sen B (Leading Span B)

    # TENKAN SAN = CONVERSION LINE. Adding the highest high and the highest low over the past nine periods and then dividing the result by two
    # KIJUN SEN = BASE LINE. Adding the highest high and the lowest low over the past 26 periods and dividing the result by two
    # SENKOU SPAN A = LEADING SPAN A. Adding the tenkan-sen and the kijun-sen, dividing the result by two, and then plotting the result 26 periods ahead
    # SENKOU SPAN B = LEADING SPAN B. Adding the highest high and the lowest low over the past 52 periods, dividing it by two, and then plotting the result 26 periods ahead

    cl_period = 9
    bl_period = 26
    senkouB_period = 2 * bl_period

    conversion_line = (df_indicators['price'].rolling(cl_period).max() + df_indicators['price'].rolling(cl_period).min()) * 0.5
    base_line = (df_indicators['price'].rolling(bl_period).max() + df_indicators['price'].rolling(bl_period).min()) * 0.5
    leadspan_a = (conversion_line + base_line) * 0.5
    leadspan_b = (df_indicators['price'].rolling(senkouB_period).max() + df_indicators['price'].rolling(senkouB_period).min()) * 0.5
    df_indicators['conversion_line'] = conversion_line
    df_indicators['base_line'] = base_line
    df_indicators['leadspan_a'] = leadspan_a
    df_indicators['leadspan_b'] = leadspan_b

    # print(df_indicators)
    return df_indicators


# def get_sma(df_prices, sym='JPM'):

#     prices = df_prices[sym]
#     prices = prices/prices[0]

#     df_indicators = pd.DataFrame(index=prices.index)
#     df_indicators['price'] = prices

#     sma = prices.rolling(10).mean()
#     df_indicators['sma'] = sma
#     ratio = df_indicators['price']/df_indicators['sma']
#     df_indicators['price_sma_ratio'] = ratio

#     return df_indicators[['price', 'price_sma_ratio']]


# def get_bollinger_bands(df_price, sym='JPM'):

#     prices = df_prices[sym]
#     prices = prices/prices[0]
#     df_indicators = pd.DataFrame(index=prices.index)
#     df_indicators['price'] = prices

#     sma = prices.rolling(10).mean()
#     df_indicators['sma'] = sma

#     stdev = df_indicators['price'].rolling(10).std()
#     two_std = 2 * stdev
#     upperband = df_indicators['sma'] + two_std
#     lowerband = df_indicators['sma'] + two_std
#     bbp = (df_indicators['price'] - df_indicators['sma']) / two_std
#     df_indicators['upper_band'] = upperband
#     df_indicators['lower_band'] = lowerband
#     df_indicators['bb_position'] = bbp

#     return df_indicators[['price', 'bbp']]


# def get_macd(df_price, sym='JPM'):
#     prices = df_prices[sym]
#     prices = prices/prices[0]
#     df_indicators = pd.DataFrame(index=prices.index)
#     df_indicators['price'] = prices

#     twelve = df_indicators['price'].ewm(span=12).mean()
#     twentysix = df_indicators['price'].ewm(span=26).mean()
#     macd = twelve - twentysix
#     signal = macd.ewm(span=9).mean()
#     diverge = macd - signal

#     df_indicators['ema_12'] = twelve
#     df_indicators['ema_26'] = twentysix
#     df_indicators['macd_value'] = macd
#     df_indicators['signal'] = signal
#     df_indicators['divergence'] = diverge

#     return df_indicators[['price', '']]

def plot_indicators():
    # Splitting Training and Test Data
    train_sd = dt.datetime(2008, 1, 1)
    train_ed = dt.datetime(2009, 12, 31)
    test_sd = dt.datetime(2010, 1, 1)
    test_ed = dt.datetime(2011, 12, 31)

    symbol = 'JPM'
    train_dates = pd.date_range(train_sd, train_ed)
    test_dates = pd.date_range(test_sd, test_ed)
    data_dates = [train_dates, test_dates]

    for i, date_range in enumerate(data_dates):
        prices_all = ut.get_data([symbol], date_range)
        inds = calculate_indicators(prices_all, symbol)
        title = ''
        if i == 0:
            title = 'IN-SAMPLE'
        else:
            title = 'OUT-OF-SAMPLE'
        # SMA
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(inds[['price', 'sma']])
        axs[0].legend(['Price', 'SMA'])
        axs[0].grid(color='grey', linestyle='-', linewidth=0.25)
        axs[0].set_ylabel('USD')
        axs[1].plot(inds[['price_sma_ratio']], color='green')
        axs[1].axhline(y=1.0, linewidth=1, linestyle='--', color='red')
        axs[1].legend(['Price/SMA'])
        axs[1].grid(color='grey', linestyle='-', linewidth=0.25)
        axs[1].set_xlabel('Datetime')
        axs[1].set_ylabel('Price/SMA')
        fig.suptitle('Price-to-SMA Ratio %s' % title)
        plt.savefig('%s_bollinger_band.png' % title)
        plt.tight_layout()
        # plt.show()

        # Bollinger Bands
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(inds[['price', 'sma', 'upper_band', 'lower_band']])
        axs[0].legend(['Price', 'SMA', 'Upper Band', 'Lower Band'])
        axs[0].grid(color='grey', linestyle='-', linewidth=0.25)
        axs[0].set_ylabel('USD')
        axs[1].plot(inds[['bb_position']])
        axs[1].axhline(y=1, linewidth=1, linestyle='--', color='red')
        axs[1].axhline(y=-1, linewidth=1, linestyle='--', color='red')
        axs[1].legend(['Bollinger Band Position'])
        axs[1].grid(color='grey', linestyle='-', linewidth=0.25)
        axs[1].set_xlabel('Datetime')
        axs[1].set_ylabel('Standard Deviations')
        fig.suptitle('Bollinger Bands %s' % title)
        plt.savefig('%s_bollinger_band.png' % title)
        plt.tight_layout()
        # plt.show()

        # MACD
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(inds[['price']])
        axs[0].legend(['Price'])
        axs[0].grid(color='grey', linestyle='-', linewidth=0.25)
        axs[0].set_ylabel('USD')
        axs[1].plot(inds[['macd_value', 'signal', 'divergence']])
        axs[1].axhline(y=0, linewidth=1, linestyle='--', color='red')
        axs[1].legend(['MACD Value', 'Signal Line', 'Divergence'])
        axs[1].grid(color='grey', linestyle='-', linewidth=0.25)
        axs[1].set_xlabel('Datetime')
        axs[1].set_ylabel('Standard Deviations')
        fig.suptitle('MACD Divergence %s' % title)
        plt.savefig('%s_bollinger_band.png' % title)
        plt.tight_layout()
        # plt.show()

        # ICHIMOKU KINKO KYO
        plt.plot(inds[['price', 'leadspan_a', 'leadspan_b']])
        plt.legend(['Price', 'Leading Span A', 'Leading Span B'])
        plt.grid(color='grey', linestyle='-', linewidth=0.25)
        plt.fill_between(inds.index, inds['leadspan_a'], inds['leadspan_b'], facecolor='pink')
        plt.ylabel('USD')
        plt.xlabel('Datetime')
        plt.title('ICHIMOKU KINKO KYO CLOUD %s' % title)
        plt.savefig('%s_bollinger_band.png' % title)
        plt.tight_layout()
        # plt.show()


def author():
    return 'jro32'


if __name__ == "__main__":
    plot_indicators()
