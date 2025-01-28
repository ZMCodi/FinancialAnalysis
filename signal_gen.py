import numpy as np
import pandas as pd

def ma_crossover(short, long):
    return np.where(short > long, 1, -1)


def rsi(RSI, ub, lb, exit, m_rev_bound=None):
    if exit == 're':
        signal = np.where(
            (RSI.shift(1) > ub) & (RSI < ub), -1, 
            np.where((RSI.shift(1) < lb) & (RSI > lb), 1, np.nan)
        )
        short_entries = (RSI.shift(1) > ub) & (RSI < ub)
    else:
        signal = np.where(
            RSI > ub, -1,
            np.where(RSI < lb, 1, np.nan)
        )
        short_entries = (RSI.shift(1) <= ub) & (RSI > ub)

    signal = pd.Series(signal, index=RSI.index)

    if np.isnan(signal.iloc[0]):
        signal.iloc[0] = 1
    signal = signal.ffill().astype(int)

    if m_rev_bound is not None:
        mean_rev_points = (RSI <= m_rev_bound) & (signal == -1)
        groups = short_entries.cumsum()
        mean_rev_triggered = mean_rev_points.groupby(groups).cummax()
        signal = np.where(mean_rev_triggered, 1, signal)

    return signal


def macd(macd_hist, rets, signal_type, combine):
    signals = pd.DataFrame(index=macd_hist.index)

    if 'crossover' in signal_type:
        signals['crossover'] = np.where(macd_hist > 0, 1, -1)

    if 'divergence' in signal_type:
        signals['divergence'] = macd_divergence(macd_hist, rets, False)

    if 'hidden divergence' in signal_type:
        signals['hidden_div'] = macd_divergence(macd_hist, rets, True)

    if 'momentum' in signal_type:
        signals['momentum'] = macd_momentum(macd_hist)

    return signals


def macd_divergence(macd_hist, rets, hidden=False):

    price_const_1 = (rets < 0)
    price_const_2 = (rets > 0)

    if hidden:
        price_const_1, price_const_2 = price_const_2, price_const_1

    signal = np.where(
        price_const_1
        & (macd_hist.shift(1) < macd_hist)
        & (macd_hist < 0), 1,
            np.where(
                price_const_2
                & (macd_hist.shift(1) > macd_hist)
                & (macd_hist.shift(1) > 0), -1, np.nan
            )
    )

    signal = pd.DataFrame(signal, index=macd_hist.index)

    if np.isnan(signal.iloc[0, 0]):
        signal.iloc[0, 0] = 1
    signal = signal.ffill().astype(int)

    return signal

def macd_momentum(macd_hist):
    signal = np.where(
        (macd_hist.shift(1) < macd_hist)
        & (macd_hist.shift(1) < 0), 1,
        np.where(
            (macd_hist.shift(1) > macd_hist)
            & (macd_hist.shift(1) > 0), -1, np.nan
        )
    )

    signal = pd.DataFrame(signal, index=macd_hist.index)

    if np.isnan(signal.iloc[0, 0]):
        signal.iloc[0, 0] = 1
    signal = signal.ffill().astype(int)

    return signal