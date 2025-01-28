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
