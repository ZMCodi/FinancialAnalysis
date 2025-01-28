import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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


def macd(macd_hist, rets, signal_type, combine, threshold, weights=None):
    signals = pd.DataFrame(index=macd_hist.index)

    if 'crossover' in signal_type:
        signals['crossover'] = np.where(macd_hist > 0, 1, -1)

    if 'divergence' in signal_type:
        signals['divergence'] = macd_divergence(macd_hist, rets, False)

    if 'hidden divergence' in signal_type:
        signals['hidden_div'] = macd_divergence(macd_hist, rets, True)

    if 'momentum' in signal_type:
        signals['momentum'] = macd_momentum(macd_hist)

    if 'double peak/trough' in signal_type:
        signals['double'] = macd_double(macd_hist)

    # combine methods: unanimous, majority, weighted
    if combine == 'unanimous':
        threshold = 1
        weights = [1 / len(signals.columns)] * len(signals.columns)
    elif combine == 'majority':
        weights = [1 / len(signals.columns)] * len(signals.columns)

    signal = vote(signals, threshold, weights)

    return signal


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


def macd_double(macd_hist):
    return double_pattern_signals(macd_hist, *find_double_patterns(macd_hist))


def find_double_patterns(macd_hist, distance_min=7, distance_max=25, prominence=0.05):
    """
    Find double tops and bottoms in MACD histogram where:
    - Double tops: Only positive peaks, first peak higher than second
    - Double bottoms: Only negative troughs, first trough lower than second

    Parameters:
    - macd_hist: MACD histogram series
    - distance_min: minimum distance between peaks
    - distance_max: maximum distance between peaks
    - prominence: required prominence of peaks
    """
    prominence *= (macd_hist.max() - macd_hist.min())

    # Find all peaks first
    peaks, _ = find_peaks(macd_hist, 
                                 distance=distance_min,
                                 prominence=prominence)

    # Filter for positive peaks only
    pos_peaks = peaks[macd_hist.iloc[peaks] > 0]

    # Find all troughs
    troughs, _ = find_peaks(-macd_hist,
                                     distance=distance_min,
                                     prominence=prominence)

    # Filter for negative troughs only
    neg_troughs = troughs[macd_hist.iloc[troughs] < 0]

    double_tops = []
    # Find double tops (first higher than second)
    for i in range(len(pos_peaks)-1):
        peak1_idx = pos_peaks[i]
        peak1_val = macd_hist.iloc[peak1_idx]

        # Look at subsequent peaks within max distance
        for j in range(i+1, len(pos_peaks)):
            peak2_idx = pos_peaks[j]
            peak2_val = macd_hist.iloc[peak2_idx]

            # Check distance constraint
            if peak2_idx - peak1_idx > distance_max:
                break

            # First peak must be higher than second
            if peak1_val > peak2_val:
                # Check for valley between peaks
                valley = macd_hist[peak1_idx:peak2_idx].min()
                if valley < peak2_val:  # Significant valley between peaks
                    double_tops.append((int(peak1_idx), int(peak2_idx)))
                    break

    double_bottoms = []
    # Find double bottoms (first lower than second)
    for i in range(len(neg_troughs)-1):
        trough1_idx = neg_troughs[i]
        trough1_val = macd_hist.iloc[trough1_idx]

        for j in range(i+1, len(neg_troughs)):
            trough2_idx = neg_troughs[j]
            trough2_val = macd_hist.iloc[trough2_idx]

            if trough2_idx - trough1_idx > distance_max:
                break

            # First trough must be lower than second
            if trough1_val < trough2_val:
                # Check for peak between troughs
                peak = macd_hist[trough1_idx:trough2_idx].max()
                if peak > trough2_val:  # Significant peak between troughs
                    double_bottoms.append((int(trough1_idx), int(trough2_idx)))
                    break

    return double_tops, double_bottoms


def double_pattern_signals(df, double_tops, double_bottoms):
    signal = pd.Series(np.nan, index=df.index)

    # Add signals at the confirmation of patterns
    for _, top2 in double_tops:
        signal.iloc[top2] = -1  # Bearish signal after second peak

    for _, bottom2 in double_bottoms:
        signal.iloc[bottom2] = 1  # Bullish signal after second trough

    if np.isnan(signal.iloc[0]):
        signal.iloc[0] = 1
    signal = signal.ffill().astype(int)

    return signal


def vote(signals, threshold, weights):
    weights = np.array(weights)

    combined = signals.dot(weights)

    signal = pd.DataFrame(np.where(combined > threshold, 1, 
                    np.where(combined < -threshold, -1, np.nan)), index=signals.index)

    if np.isnan(signal.iloc[0, 0]):
        signal.iloc[0, 0] = 1
    signal = signal.ffill().astype(int)

    return signal