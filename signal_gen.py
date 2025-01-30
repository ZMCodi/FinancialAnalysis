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

    signal = fill(signal)

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

    signal = pd.Series(signal, index=macd_hist.index)

    return fill(signal)

def macd_momentum(macd_hist):
    signal = np.where(
        (macd_hist.shift(1) < macd_hist)
        & (macd_hist.shift(1) < 0), 1,
        np.where(
            (macd_hist.shift(1) > macd_hist)
            & (macd_hist.shift(1) > 0), -1, np.nan
        )
    )

    signal = pd.Series(signal, index=macd_hist.index)

    return fill(signal)


def macd_double(macd_hist):
    return double_pattern_signals(macd_hist, *find_double_patterns(macd_hist))


def bb(price, bb_up, bb_down, signal_type, combine, threshold, weights=None):
    signals = pd.DataFrame(index=price.index)

    if 'bounce' in signal_type:
        signals['bounce'] = bb_bounce(price, bb_up, bb_down)

    if 'double' in signal_type:
        signals['double'] = bb_double(price, bb_up, bb_down)

    if 'walks' in signal_type:
        signals['walks'] = bb_walks(price, bb_up, bb_down)

    if 'squeeze' in signal_type:
        signals['squeeze'] = bb_squeeze(price, bb_up, bb_down)

    if 'breakout' in signal_type:
        signals['breakout'] = bb_breakout(price, bb_up, bb_down)

    if '%B' in signal_type:
        signals['%B'] = bb_pctB(price, bb_up, bb_down)

    # combine methods: unanimous, majority, weighted
    if combine == 'unanimous':
        threshold = 1
        weights = [1 / len(signals.columns)] * len(signals.columns)
    elif combine == 'majority':
        weights = [1 / len(signals.columns)] * len(signals.columns)

    signal = vote(signals, threshold, weights)

    return signal


def bb_bounce(price, bb_up, bb_down):
    signal = pd.Series(np.where(
        (price.shift(1) > bb_up) & (price < bb_up), -1,
        np.where(
            (price.shift(1) < bb_down) & (price > bb_down), 1, np.nan
        )
    ), index=price.index)

    return fill(signal)


def bb_double(price, bb_up, bb_down):
    rel_width = bb_up - bb_down
    hist = pd.Series(np.where(
        price > bb_up, (price - bb_up) / rel_width,
        np.where(price < bb_down, (price - bb_down) / rel_width, 0)
    ), index=price.index)

    return double_pattern_signals(price, *find_double_patterns(hist, 5, 15))


def bb_walks(price, bb_up, bb_down, prox=0.2, periods=5):
    width = bb_up - bb_down
    close_upper = np.abs(price - bb_up) < width * prox
    close_lower = np.abs(price - bb_down) < width * prox

    upper_walk = close_upper.rolling(periods).sum() >= periods - 1
    lower_walk = close_lower.rolling(periods).sum() >= periods - 1

    walk = pd.Series(np.where(upper_walk, 1,
                    np.where(lower_walk, -1, np.nan)), index=price.index)

    return fill(walk)

def bb_squeeze(price, bb_up, bb_down, aggressive=False):
    width = bb_up - bb_down
    squeeze = width < width.rolling(20).quantile(0.2)
    
    if aggressive:
        ext = (width > width.shift(1)) & squeeze.shift(1)
    else:
        ext = ~squeeze & squeeze.shift(1)

    signal = pd.Series(np.where(
        ext & (price > price.shift(1)), 1,
        np.where(ext & (price < price.shift(1)), -1, np.nan)
    ), index=price.index)

    return fill(signal)


def bb_breakout(price, bb_up, bb_down, threshold=0.3):
    momentum = price.pct_change()
    mom_range = momentum.max() - momentum.min()

    signal = pd.Series(np.where(
        (price > bb_up) & (momentum > threshold * mom_range), 1,
        np.where((price < bb_down) & (momentum < -threshold * mom_range), -1, np.nan)
    ), index=price.index)

    return fill(signal)


def bb_pctB(price, bb_up, bb_down, overbought=0.8, oversold=0.2):
    pctB = (price - bb_down) / (bb_up - bb_down)
    signal = pd.Series(np.where(pctB > overbought, -1,
                      np.where(pctB < oversold, 1, np.nan)), index=price.index)

    return fill(signal)


def find_double_patterns(hist, distance_min=7, distance_max=25, prominence=0.05):
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
    prominence *= (hist.max() - hist.min())

    # Find all peaks first
    peaks, _ = find_peaks(hist, 
                          distance=distance_min,
                          prominence=prominence)

    # Filter for positive peaks only
    pos_peaks = peaks[hist.iloc[peaks] > 0]

    # Find all troughs
    troughs, _ = find_peaks(-hist,
                            distance=distance_min,
                            prominence=prominence)

    # Filter for negative troughs only
    neg_troughs = troughs[hist.iloc[troughs] < 0]

    double_tops = []
    # Find double tops (first higher than second)
    for i in range(len(pos_peaks)-1):
        peak1_idx = pos_peaks[i]
        peak1_val = hist.iloc[peak1_idx]

        # Look at subsequent peaks within max distance
        for j in range(i+1, len(pos_peaks)):
            peak2_idx = pos_peaks[j]
            peak2_val = hist.iloc[peak2_idx]

            # Check distance constraint
            if peak2_idx - peak1_idx > distance_max:
                break

            # First peak must be higher than second
            if peak1_val > peak2_val:
                # Check for valley between peaks
                valley = hist[peak1_idx:peak2_idx].min()
                if valley < peak2_val:  # Significant valley between peaks
                    double_tops.append((int(peak1_idx), int(peak2_idx)))
                    break

    double_bottoms = []
    # Find double bottoms (first lower than second)
    for i in range(len(neg_troughs)-1):
        trough1_idx = neg_troughs[i]
        trough1_val = hist.iloc[trough1_idx]

        for j in range(i+1, len(neg_troughs)):
            trough2_idx = neg_troughs[j]
            trough2_val = hist.iloc[trough2_idx]

            if trough2_idx - trough1_idx > distance_max:
                break

            # First trough must be lower than second
            if trough1_val < trough2_val:
                # Check for peak between troughs
                peak = hist[trough1_idx:trough2_idx].max()
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

    return fill(signal)


def vote(signals, threshold, weights):
    weights = np.array(weights)

    combined = signals.dot(weights)

    signal = pd.Series(np.where(combined > threshold, 1, 
                    np.where(combined < -threshold, -1, np.nan)), index=signals.index)

    return fill(signal)

def fill(series, default=1):
    if np.isnan(series.iloc[0]):
        series.iloc[0] = default

    return series.ffill().astype(int)

# TODO:
# bb signals: bounce, double, walks, squeeze, breakout, pctB
# better divergence method
# more rsi signals
# ATR, ADX