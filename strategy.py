''' Strategy module for implementing and backtesting technical analysis trading strategies

Key components:
- TAEngine: Technical analysis calculation engine with caching for efficient computation
- Strategy base class with plotting, backtesting and optimization capabilities
- Concrete strategy implementations:
    - Moving Average Crossover
    - RSI with multiple signal types
    - MACD with multiple signal types  
    - Bollinger Bands with multiple signal types
    - Combined strategy approach with signal voting

Features:
- Strategy visualization with interactive Plotly charts
- Backtesting with customizable parameters and timeframes
- Parameter optimization via grid search
- Strategy weights optimization
- Signal combination through weighted voting
'''

from abc import ABC, abstractmethod
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import signal_gen as sg
import scipy.optimize as sco
from assets import Asset
from typing import Optional
from datetime import datetime, date

DateLike = str | datetime | date | pd.Timestamp

class TAEngine:
    """Technical Analysis calculation engine with caching capabilities.
    
    A utility class that provides methods to calculate common technical indicators
    while caching results to avoid redundant computations. Can be used standalone
    or as part of strategy implementations.

    The engine caches results using a combination of parameters and data name
    as keys, allowing efficient reuse when the same calculations are needed
    multiple times, such as during optimization.

    Attributes:
        cache (dict): Dictionary storing computed technical indicators
    """

    def __init__(self):
        """Initialize an empty cache for storing computed indicators."""
        self.cache = {}

    def calculate_ma(self, data: pd.Series, ewm: bool, param_type: str, 
                    param: float, name: str) -> pd.Series:
        """Calculate moving average with caching.

        Computes either simple moving average or exponential weighted moving
        average based on specified parameters. Results are cached using a key
        that combines all parameters.

        Args:
            data (pd.Series): Price series to calculate MA for
            ewm (bool): If True, use exponential weighted moving average.
                If False, use simple moving average
            param_type (str): Parameter type for the moving average.
                Can be 'window', 'span', 'com', 'halflife', or 'alpha'
            param (float): Value for the specified parameter type
            name (str): Identifier for the data series, used in cache key

        Returns:
            pd.Series: Moving average series
        """
        key = f'ma_{param_type}={param}, {name=}'
        if key not in self.cache:
            if ewm:
                self.cache[key] = data.ewm(**{f'{param_type}': param}).mean()
            else:
                self.cache[key] = data.rolling(window=param).mean()

        return self.cache[key]

    def calculate_rsi(self, data: pd.Series, window: int, name: str) -> pd.Series:
        """Calculate Relative Strength Index (RSI) with caching.

        Computes RSI using the standard formula with exponential weighted
        moving averages for the gains and losses.

        Args:
            data (pd.Series): Price series to calculate RSI for
            window (int): Lookback period for RSI calculation
            name (str): Identifier for the data series, used in cache key

        Returns:
            pd.Series: RSI values ranging from 0 to 100
        """
        key = f'rsi_window={window}, {name=}'
        if key not in self.cache:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
            rs = gain / loss
            self.cache[key] = 100 - (100 / (1 + rs))

        return self.cache[key]

    def calculate_macd(self, data: pd.Series, windows: list[int], name: str) -> pd.DataFrame:
        """Calculate Moving Average Convergence Divergence (MACD) with caching.

        Computes MACD using three exponential moving averages:
        - Fast EMA
        - Slow EMA 
        - Signal line EMA on the MACD line

        The function converts the window periods to alpha values and caches
        the results including the MACD line, signal line, and histogram.

        Args:
            data (pd.Series): Price series to calculate MACD for
            windows (list[int]): Three window periods [fast, slow, signal]
                where fast < slow and signal is for the MACD line
            name (str): Identifier for the data series, used in cache key

        Returns:
            pd.DataFrame: DataFrame with columns:
                - macd: The MACD line (fast EMA - slow EMA)
                - signal_line: EMA of the MACD line
                - macd_hist: MACD histogram (signal_line - macd)
        """
        key = f'macd_{windows=}, {name=}'
        if key not in self.cache:
            results = pd.DataFrame(index=data.index)

            alpha_fast = 2 / (windows[0] + 1)
            alpha_slow = 2 / (windows[1] + 1)
            alpha_signal = 2 / (windows[2] + 1)

            results['macd'] = (self.calculate_ma(data, True, 'alpha', alpha_fast, name)
                                - self.calculate_ma(data, True, 'alpha', alpha_slow, name))

            results['signal_line'] = self.calculate_ma(results['macd'], True, 'alpha', alpha_signal, name)

            results['macd_hist'] = results['signal_line'] - results['macd']

            self.cache[key] = results

        return self.cache[key]

    def rolling_std(self, data: pd.Series, ewm: bool, param_type: str, 
                    param: float, name: str) -> pd.Series:
        """Calculate rolling standard deviation with caching.

        Computes either simple rolling standard deviation or exponential weighted
        standard deviation based on specified parameters.

        Args:
            data (pd.Series): Price series to calculate standard deviation for
            ewm (bool): If True, use exponential weighted standard deviation.
                If False, use simple rolling standard deviation
            param_type (str): Parameter type for the calculation.
                Can be 'window', 'span', 'com', 'halflife', or 'alpha'
            param (float): Value for the specified parameter type
            name (str): Identifier for the data series, used in cache key

        Returns:
            pd.Series: Rolling standard deviation series
        """
        key = f'rol_std_{param_type}={param}, {name=}'
        if key not in self.cache:
            if ewm:
                self.cache[key] = data.ewm(**{f'{param_type}': param}).std()
            else:
                self.cache[key] = data.rolling(window=param).std()

        return self.cache[key]

    def calculate_bb(self, data: pd.Series, window: int, num_std: float, 
                    name: str) -> pd.DataFrame:
        """Calculate Bollinger Bands with caching.

        Computes Bollinger Bands using:
        - Simple moving average (middle band)
        - Upper band: SMA + (standard deviation × num_std)
        - Lower band: SMA - (standard deviation × num_std)

        Args:
            data (pd.Series): Price series to calculate Bollinger Bands for
            window (int): Window period for the moving average and standard deviation
            num_std (float): Number of standard deviations for the bands
            name (str): Identifier for the data series, used in cache key

        Returns:
            pd.DataFrame: DataFrame with columns:
                - sma: Simple moving average (middle band)
                - bol_up: Upper Bollinger Band
                - bol_down: Lower Bollinger Band
        """
        key = f'bb_{window=}_{num_std=}, {name}'
        if key not in self.cache:
            results = pd.DataFrame(index=data.index)

            results['sma'] = self.calculate_ma(data, False, 'window', window, name)
            std = self.rolling_std(data, False, 'window', window, name)
            results['bol_up'] = results['sma'] + num_std * std
            results['bol_down'] = results['sma'] - num_std * std

            self.cache[key] = results

        return self.cache[key]


class Strategy(ABC):
    """Abstract base class for implementing trading strategies.
    
    Provides a framework for creating technical analysis trading strategies with
    built-in backtesting, visualization, and optimization capabilities. All concrete
    strategy implementations should inherit from this class and implement the
    abstract methods.

    The class handles both daily and 5-minute data timeframes, supports parameter
    optimization through grid search, and allows strategy visualization through
    Plotly interactive charts.

    Attributes:
        asset (Asset): Asset object containing price data and metadata
        daily (pd.DataFrame): DataFrame containing daily trading signals and returns
        five_min (pd.DataFrame): DataFrame containing 5-minute trading signals and returns
        params (str): String representation of strategy parameters

    Abstract Methods:
        plot: Visualize the strategy and its signals
        optimize: Optimize strategy parameters using grid search
    """

    def __init__(self, asset: Asset):
        """Initialize the strategy for a given asset.

        Args:
            asset (Asset): Asset object containing price data and metadata
        """
        self.asset = asset

    @abstractmethod
    def plot(self):
        """Plot strategy indicators and signals.
        
        To be implemented by concrete strategy classes.
        Should create visualization showing strategy indicators and generated signals.
        """
        pass

    @abstractmethod
    def optimize(self):
        """Optimize strategy parameters.
        
        To be implemented by concrete strategy classes.
        Should perform grid search over parameter space to find optimal settings.
        """
        pass

    def backtest(self, plot: bool = True, timeframe: str = '1d', 
                start_date: Optional[DateLike] = None, end_date: Optional[DateLike] = None, 
                show_signal: bool = True, fig: Optional[go.Figure] = None, 
                subplot_idx: Optional[tuple[int, int]] = None) -> pd.Series:
        """Backtest the strategy and optionally plot results.

        Performs backtesting by applying the strategy's signals to historical data
        and calculating cumulative returns. Can visualize the results using an
        interactive Plotly chart showing both buy-and-hold and strategy returns.

        Args:
            plot (bool, optional): Whether to create visualization. Defaults to True.
            timeframe (str, optional): Data frequency to use ('1d' or '5m'). Defaults to '1d'.
            start_date (DateLike, optional): Start date for backtest. Defaults to None.
            end_date (DateLike, optional): End date for backtest. Defaults to None.
            show_signal (bool, optional): Whether to plot signals. Defaults to True.
            fig (go.Figure, optional): Existing figure to add plots to. Defaults to None.
            subplot_idx (tuple[int, int], optional): Subplot position (row, col). Defaults to None.

        Returns:
            pd.Series: Series with two values:
                - returns: Buy-and-hold cumulative returns
                - strategy: Strategy cumulative returns
        """
        name = self.__class__.__name__
        df = self.daily if timeframe == '1d' else self.five_min
        df.dropna(inplace=True)

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        if plot:
            trace1 = go.Scatter(
                x=df.index,
                y=np.exp(df['returns'].cumsum()),
                line=dict(
                    color='#2962FF',
                    width=2,
                    dash='solid'
                ),
                name=f'{self.asset.ticker} Hold Returns',
                yaxis='y'
            )

            trace2 = go.Scatter(
                x=df.index,
                y=np.exp(df['strategy'].cumsum()),
                line=dict(
                    color='red',
                    width=2,
                    dash='solid'
                ),
                name=f'{self.asset.ticker} Strategy Returns',
                yaxis='y'
            )

            if show_signal:
                trace3 = go.Scatter(
                    x=df.index,
                    y=df['signal'],
                    line=dict(color='green', width=0.8, dash='solid'),
                    name='Buy/Sell signal',
                    yaxis='y2'
                )

            # Add traces based on whether it's a subplot or not
            standalone = False
            if fig is None:
                standalone = True
                fig = go.Figure()

            fig.add_trace(trace1,
                        row=subplot_idx[0] if subplot_idx else None,
                        col=subplot_idx[1] if subplot_idx else None)

            fig.add_trace(trace2, 
                        row=subplot_idx[0] if subplot_idx else None,
                        col=subplot_idx[1] if subplot_idx else None)

            if show_signal:
                if standalone:
                    fig.add_trace(trace3)
                else:
                    fig.add_trace(trace3,
                                row=subplot_idx[0] if subplot_idx else None,
                                col=subplot_idx[1] if subplot_idx else None,
                                secondary_y=True)

            # Update layout with secondary y-axis
            layout = {}

            if standalone:
                layout['title'] = dict(
                        text=f'{self.asset.ticker} {name} Backtest {self.params}',
                        x=0.5,
                        y=0.95
                    )

            layout['xaxis'] = dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    title=None,
                )

            layout['yaxis'] = dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    title=f'Returns',
                )

            if show_signal:
                layout['yaxis2'] = dict(
                        title='Signal',
                        overlaying='y',
                        side='right',
                        range=[-1.1, 1.1],
                        tickmode='array',
                        tickvals=[-1, 1],
                        ticktext=['Sell', 'Buy']
                    )

            layout['legend'] = dict(
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    orientation="h",
                    bgcolor='rgba(255,255,255,0.8)'
                )

            fig.update_layout(**layout,
                                paper_bgcolor='white',
                                plot_bgcolor='rgba(240,240,240,0.95)',
                                hovermode='x unified')

            if standalone:
                fig.show()
            else:
                fig.update_yaxes(
                    title_text=f'Returns',
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )
                fig.update_xaxes(
                    title_text=f'{self.asset.ticker} {name} Backtest {self.params}', 
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )

        return np.exp(df[['returns', 'strategy']].sum())
    
    def optimize_weights(self, inplace: bool = False, timeframe: str = '1d',
                        start_date: Optional[DateLike] = None, end_date: Optional[DateLike] = None, 
                        threshold_range: Optional[np.ndarray] = None, runs: int = 10) -> tuple[np.ndarray, float]:
        """Optimize signal combination weights and voting threshold.

        Uses scipy's SLSQP optimizer to find optimal weights for combining multiple
        signals from the strategy. Includes regularization to prevent extreme weights
        and entropy bonus to encourage weight diversity.

        Args:
            inplace (bool, optional): Whether to update strategy weights. Defaults to False.
            timeframe (str, optional): Data frequency to use ('1d' or '5m'). Defaults to '1d'.
            start_date (DateLike, optional): Start date for optimization. Defaults to None.
            end_date (DateLike, optional): End date for optimization. Defaults to None.
            threshold_range (np.ndarray, optional): Range of voting thresholds to test. 
                Defaults to np.arange(0.2, 0.9, 0.1).
            runs (int, optional): Number of random initializations. Defaults to 10.

        Returns:
            tuple[np.ndarray, float]: Tuple containing:
                - Optimal weights for each signal
                - Optimal voting threshold
        """
        old_params = {'weights': self.weights, 'vote_threshold': self.vote_threshold}

        def objective_function(params):
            weights = params[:-1]
            threshold = params[-1]

            # Now evaluate combined signal with given weights
            self.change_params(weights=weights, vote_threshold=threshold)
            combined_returns = self.backtest(plot=False,
                                        timeframe=timeframe,
                                        start_date=start_date,
                                        end_date=end_date)['strategy']

            # Add regularization terms to prevent extreme weights
            diversity_bonus = 0.1 * np.sum(-weights * np.log(weights + 1e-10))  # Entropy term
            extreme_penalty = 0.05 * np.sum(weights ** 2)  # L2 regularization

            return -combined_returns - diversity_bonus + extreme_penalty

        n_weights = len(self.weights)
        if threshold_range is None:
            threshold_range = np.arange(0.2, 0.9, 0.1)

        t_min, t_max = threshold_range[0], threshold_range[-1]

        cons = ({
            'type': 'eq',
            'fun': lambda x: np.sum(x[:-1]) - 1
        })

        bnds = tuple([(0, 1)] * n_weights + [(t_min, t_max)])

        # Try multiple random initializations
        best_result = None
        best_value = float('inf')

        for _ in range(runs):
            # Random initial weights that sum to 1
            init_weights = np.random.dirichlet(np.ones(n_weights))
            init_threshold = np.random.uniform(t_min, t_max)
            init_params = np.concatenate([init_weights, [init_threshold]])

            result = sco.minimize(objective_function, init_params,
                                method='SLSQP', bounds=bnds,
                                constraints=cons)

            if result.fun < best_value:
                best_value = result.fun
                best_result = result

        # Split results
        opt_weights = best_result.x[:-1] / np.sum(best_result.x[:-1])
        opt_threshold = best_result.x[-1]

        if inplace:
            self.change_params(weights=opt_weights, vote_threshold=opt_threshold)
        else:
            self.change_params(**old_params)

        return opt_weights, float(opt_threshold)

    @property 
    def num_signals_daily(self) -> int:
        """Count the number of signal changes in daily data.

        Returns:
            int: Number of times the trading signal changes from -1 to 1 or vice versa
        """
        return np.sum(np.where(self.daily['signal'].shift(1) != self.daily['signal'], 1, 0))

    @property
    def num_signals_five_min(self) -> int:
        """Count the number of signal changes in 5-minute data.

        Returns:
            int: Number of times the trading signal changes from -1 to 1 or vice versa
        """
        return np.sum(np.where(self.five_min['signal'].shift(1) != self.five_min['signal'], 1, 0))


class MA_Crossover(Strategy):

    def __init__(self, asset, param_type='window', short_window=20, long_window=50,
                short_alpha=None, long_alpha=None, short_halflife=None, long_halflife=None, ewm=False):

        super().__init__(asset)
        self.ptype = param_type
        self.ewm = ewm
        self.__short = eval(f'short_{param_type}')
        self.__long = eval(f'long_{param_type}')
        self.engine = TAEngine()
        self.__get_data()

    def __str__(self):
        return f'MA_Crossover({self.asset.ticker}, short_{self.ptype}={self.short}, long_{self.ptype}={self.long})'

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily[['adj_close', 'log_rets']])
        self.five_min = pd.DataFrame(self.asset.five_minute[['adj_close', 'log_rets']])
        self.params = f'({self.short}/{self.long})'

        for i, df in enumerate([self.daily, self.five_min]):
            data = df['adj_close']
            name = 'daily' if i == 0 else 'five_min'
            ptype = 'span' if self.ptype == 'window' and self.ewm else self.ptype

            df['short'] = self.engine.calculate_ma(data, self.ewm, ptype, self.short, name)
            df['long'] = self.engine.calculate_ma(data, self.ewm, ptype, self.long, name)
            df.dropna(inplace=True)

            df['signal'] = sg.ma_crossover(df['short'], df['long'])
            df.rename(columns=dict(log_rets='returns'), inplace=True)
            df['strategy'] = df['returns'] * df['signal']
            if i == 0:
                self.daily = df
            else:
                self.five_min = df

    @property
    def short(self):
        return self.__short

    @short.setter
    def short(self, value):
        self.__short = value
        self.__get_data()

    @property
    def long(self):
        return self.__long

    @long.setter
    def long(self, value):
        self.__long = value
        self.__get_data()

    def change_params(self, param_type=None, short=None, long=None, ewm=None):
        self.ptype = param_type if param_type is not None else self.ptype
        self.__short = short if short is not None else self.short
        self.__long = long if long is not None else self.long
        self.ewm = ewm if ewm is not None else self.ewm
        self.__get_data()

    def plot(self, timeframe='1d', start_date=None, end_date=None, 
            show_signal=True):

        df = self.daily if timeframe == '1d' else self.five_min

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        df.dropna(inplace=True)

        long_data = df['long']
        short_data = df['short']

        if show_signal:
            signal = df['signal']

        short_param = f'{self.ptype}={self.short}'
        long_param = f'{self.ptype}={self.long}'

        # Add short MA line
        short_MA = go.Scatter(
            x=short_data.index,
            y=short_data,
            line=dict(
                color='#2962FF',
                width=2,
                dash='solid'
            ),
            name=f'{self.asset.ticker} MA ({short_param})',
            yaxis='y'
        )

        # Add long MA line
        long_MA = go.Scatter(
            x=long_data.index,
            y=long_data,
            line=dict(
                color='red',
                width=2,
                dash='solid'
            ),
            name=f'{self.asset.ticker} MA ({long_param})',
            yaxis='y'
        )

        if show_signal:
            signal = go.Scatter(
                x=signal.index,
                y=signal,
                line=dict(color='green', width=0.8, dash='solid'),
                name='Buy/Sell signal',
                yaxis='y2'
            )

        # Add traces based on whether it's a subplot or not
        fig = go.Figure()

        fig.add_trace(short_MA)

        fig.add_trace(long_MA)

        if show_signal:
            fig.add_trace(signal)

        # Update layout with secondary y-axis
        layout = {}

        layout['title'] = dict(
                text=f'{self.asset.ticker} MA Crossover {self.params}',
                x=0.5,
                y=0.95
            )

        layout['xaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=None,
            )

        layout['yaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=f'Price ({self.asset.currency})',
            )

        if show_signal:
            layout['yaxis2'] = dict(
                    title='Signal',
                    overlaying='y',
                    side='right',
                    range=[-1.1, 1.1],  # Give some padding to the 0/1 signal
                    tickmode='array',
                    tickvals=[-1, 1],
                    ticktext=['Sell', 'Buy']
                )

        layout['legend'] = dict(
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                orientation="h",  # horizontal layout
                bgcolor='rgba(255,255,255,0.8)'
            )

        fig.update_layout(**layout,
                            paper_bgcolor='white',
                            plot_bgcolor='rgba(240,240,240,0.95)',
                            hovermode='x unified')


        fig.show()

        return fig

    def optimize(self, inplace=False, timeframe='1d', start_date=None, end_date=None,
                 short_range=None, long_range=None):

        if short_range is None:
            if self.ptype == 'window':
                short_range = np.arange(20, 61, 5)  # window
            else:
                short_range = np.arange(0.10, 0.31, 0.03)  # alpha
                if self.ptype == 'halflife':
                    short_range = -np.log(2) / np.log(1 - short_range)  # halflife

        if long_range is None:
            if self.ptype == 'window':
                long_range = np.arange(100, 281, 10)
            else:
                long_range = np.arange(0.01, 0.11, 0.02)  # alpha
                if self.ptype == 'halflife':
                    long_range = -np.log(2) / np.log(1 - long_range)  # halflife

        old_params = {'short': self.short, 'long': self.long, 'ewm': self.ewm, 'param_type': self.ptype}

        results = []
        for short, long in product(short_range, long_range):
            if self.ptype == 'alpha' and short <= long:
                continue
            elif short >= long:
                continue

            self.change_params(short=short, long=long)
            backtest_results = self.backtest(plot=False, 
                                        timeframe=timeframe, 
                                        start_date=start_date,
                                        end_date=end_date)
            results.append((short, long, backtest_results['returns'], backtest_results['strategy']))

        results = pd.DataFrame(results, columns=['short', 'long', 'hold_returns', 'strategy_returns'])
        results['net'] = results['strategy_returns'] - results['hold_returns']
        results = results.sort_values(by='net', ascending=False)

        opt_short = results.iloc[0]['short']
        opt_long = results.iloc[0]['long']

        if self.ptype == 'window':
            opt_short = int(opt_short)
            opt_long = int(opt_long)

        if inplace:
            self.change_params(short=opt_short, long=opt_long)
        else:
            self.change_params(**old_params)

        return results
    
    def optimize_weights(self):
        raise NotImplementedError("No weights associated with this strategy")


class RSI(Strategy):

    def __init__(self, asset, ub=70, lb=30, window=14, exit='re', m_rev=True, m_rev_bound=50,
                 signal_type=None, combine='weighted', weights=None, vote_threshold=0.5,):

        super().__init__(asset)
        self.__ub = ub
        self.__lb = lb
        self.__window = window
        self.__exit = exit
        self.__m_rev = m_rev
        self.__m_rev_bound = m_rev_bound
        if signal_type is not None:
            self.signal_type = list(signal_type)
        else:
            self.signal_type = ['crossover', 'divergence', 'hidden divergence']

        self.__combine = str(combine)

        if weights is not None:
            self.__weights = np.array(weights)
        else:
            self.__weights = np.array([1 / len(self.signal_type)] * len(self.signal_type))
        self.__weights /= np.sum(self.__weights)

        self.__vote_threshold = vote_threshold
        self.engine = TAEngine()
        self.__get_data()

    def __str__(self):
        return f'RSI({self.asset.ticker}, ub={self.ub}, lb={self.lb}, window={self.window}, exit={self.exit}, m_rev={self.m_rev}, m_rev_bound={self.m_rev_bound})'

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])
        self.five_min = pd.DataFrame(self.asset.five_minute[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])

        self.params = f'({self.ub}/{self.lb})'

        for i, df in enumerate([self.daily, self.five_min]):
            data = df['adj_close']
            name = 'daily' if i == 0 else 'five_min'

            df['rsi'] = self.engine.calculate_rsi(data, self.window, name)
            df.dropna(inplace=True)

            df['signal'] = sg.rsi(df['rsi'], df['adj_close'], self.ub, self.lb, self.exit, self.signal_type,
                            self.combine, self.vote_threshold, self.weights, self.m_rev_bound if self.m_rev else None)

            df.rename(columns=dict(log_rets='returns'), inplace=True)
            df['strategy'] = df['returns'] * df['signal']

            if i == 0:
                self.daily = df
            else:
                self.five_min = df

    @property
    def ub(self):
        return self.__ub

    @ub.setter
    def ub(self, value):
        self.__ub = value
        self.__get_data()

    @property
    def lb(self):
        return self.__lb

    @lb.setter
    def lb(self, value):
        self.__lb = value
        self.__get_data()

    @property
    def m_rev(self):
        return self.__m_rev

    @m_rev.setter
    def m_rev(self, value):
        self.__m_rev = value
        self.__get_data()

    @property
    def exit(self):
        return self.__exit

    @exit.setter
    def exit(self, value):
        self.__exit = value
        self.__get_data()

    @property
    def window(self):
        return self.__window

    @window.setter
    def window(self, value):
        self.__window = value
        self.__get_data()

    @property
    def m_rev_bound(self):
        return self.__m_rev_bound

    @m_rev_bound.setter
    def m_rev_bound(self, value):
        self.__m_rev_bound = value
        self.__get_data()

    @property
    def combine(self):
        return self.__combine

    @combine.setter
    def combine(self, value):
        self.__combine = value
        self.__get_data()

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = np.array(value)
        self.__weights /= np.sum(self.__weights)
        self.__get_data()

    @property
    def vote_threshold(self):
        return self.__vote_threshold

    @vote_threshold.setter
    def vote_threshold(self, value):
        self.__vote_threshold = value
        self.__get_data()

    def change_params(self, ub=None, lb=None, window=None, exit=None, m_rev=None, m_rev_bound=None,
                      combine=None, weights=None, vote_threshold=None):
        self.__ub = ub if ub is not None else self.ub
        self.__lb = lb if lb is not None else self.lb
        self.__window = window if window is not None else self.window
        self.__exit = exit if exit is not None else self.exit
        self.__m_rev = m_rev if m_rev is not None else self.m_rev
        self.__m_rev_bound = m_rev_bound if m_rev_bound is not None else self.m_rev_bound
        self.__weights = np.array(weights) if weights is not None else self.weights
        self.__weights /= np.sum(self.__weights)
        self.__vote_threshold = vote_threshold if vote_threshold is not None else self.vote_threshold
        self.__get_data()

    def plot(self, timeframe='1d', start_date=None, end_date=None,
            candlestick=True, show_signal=True):

        df = self.daily if timeframe == '1d' else self.five_min

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        df.dropna(inplace=True)

        fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        specs=[[{"secondary_y": True}],
                        [{"secondary_y": False}]])

        if candlestick:
            price = go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f'{self.asset.ticker} OHLC',
            )
        else:
            price = go.Scatter(
                        x=df.index,
                        y=df['adj_close'],
                        line=dict(
                            color='#2962FF',
                            width=2,
                            dash='solid'
                        ),
                        name=f'{self.asset.ticker} Price',
            )

        RSI = go.Scatter(
                x=df.index,
                y=df['rsi'],
                line=dict(color='blue', width=1.5),
                name='RSI'
        )

        if show_signal:
            signal = go.Scatter(
                        x=df.index,
                        y=df['signal'],
                        line=dict(color='green', width=0.8, dash='solid'),
                        name='Buy/Sell signal',
                        yaxis='y2'
            )

        fig.add_trace(price)

        fig.add_trace(RSI, row=2, col=1)
        fig.add_hline(y=self.ub, row=2, col=1)
        fig.add_hline(y=self.lb, row=2, col=1)

        if show_signal:
            fig.add_trace(signal)

        layout = {}

        layout['title'] = dict(
                text=f'{self.asset.ticker} RSI Strategy {self.params}',
                x=0.5,
                y=0.95
            )

        layout['xaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=None,
            )

        layout['yaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=f'Price ({self.asset.currency})',
            )

        layout[f'xaxis1_rangeslider_visible'] = False

        layout['height'] = 800

        if show_signal:
            layout['yaxis2'] = dict(
                    title='Signal',
                    overlaying='y',
                    side='right',
                    range=[-1.1, 1.1],  # Give some padding to the 0/1 signal
                    tickmode='array',
                    tickvals=[-1, 1],
                    ticktext=['Sell', 'Buy']
                )

        layout['yaxis3'] = dict(
                title='RSI',
                side='right'
            )

        layout['legend'] = dict(
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                orientation="h",  # horizontal layout
                bgcolor='rgba(255,255,255,0.8)'
            )

        fig.update_layout(**layout,
                            paper_bgcolor='white',
                            plot_bgcolor='rgba(240,240,240,0.95)',
                            hovermode='x unified')

        fig.show()

        return fig

    def optimize(self, inplace=False, timeframe='1d', start_date=None, end_date=None,
                ub_range=None, lb_range=None, window_range=None, m_rev_bound_range=None):

        if ub_range is None:
            ub_range = np.arange(60, 81, 5)
        if lb_range is None:
            lb_range = np.arange(20, 41, 5)
        if window_range is None:
            window_range = np.arange(10, 31, 5)

        params = [ub_range, lb_range, window_range]

        if self.m_rev:
            if m_rev_bound_range is None:
                m_rev_bound_range = np.arange(40, 61, 5)
            params.append(m_rev_bound_range)
        else:
            params.append([self.m_rev_bound])

        old_params = {'ub': self.ub, 'lb': self.lb, 'window': self.window,
                      'm_rev_bound': self.m_rev_bound}

        results = []
        for ub, lb, window, m_rev_bound in product(*params):
            if ub <= lb or m_rev_bound >= ub or m_rev_bound <= lb:
                continue

            self.change_params(ub=ub, lb=lb, window=window, m_rev_bound=m_rev_bound)
            backtest_results = self.backtest(plot=False, 
                                        timeframe=timeframe, 
                                        start_date=start_date,
                                        end_date=end_date)
            results.append((ub, lb, window, m_rev_bound, backtest_results['returns'], backtest_results['strategy']))

        results = pd.DataFrame(results, columns=['ub', 'lb', 'window', 'm_rev_bound', 'hold_returns', 'strategy_returns'])
        results['net'] = results['strategy_returns'] - results['hold_returns']
        results = results.sort_values(by='net', ascending=False)

        opt_ub = results.iloc[0]['ub']
        opt_lb = results.iloc[0]['lb']
        opt_window = results.iloc[0]['window']
        opt_m_rev_bound = results.iloc[0]['m_rev_bound']

        if inplace:
            self.change_params(ub=opt_ub, lb=opt_lb, window=opt_window, m_rev_bound=opt_m_rev_bound)
        else:
            self.change_params(**old_params)

        return results


class MACD(Strategy):

    def __init__(self, asset, fast=12, slow=26, signal=9, signal_type=None, combine='weighted', weights=None, vote_threshold=0.5):
        super().__init__(asset)
        self.__slow = slow
        self.__fast = fast
        self.__signal = signal

        if signal_type is not None:
            self.signal_type = list(signal_type)
        else:
            self.signal_type = ['crossover', 'divergence', 'hidden divergence', 'momentum', 'double peak/trough']

        self.__combine = str(combine)

        if weights is not None:
            self.__weights = np.array(weights)
        else:
            self.__weights = np.array([1 / len(self.signal_type)] * len(self.signal_type))
        self.__weights /= np.sum(self.__weights)

        self.__vote_threshold = vote_threshold

        self.engine = TAEngine()

        self.__get_data()

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])
        self.five_min = pd.DataFrame(self.asset.five_minute[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])

        self.params = f'({self.fast}/{self.slow}/{self.signal})'

        for i, df in enumerate([self.daily, self.five_min]):
            data = df['adj_close']
            name = 'daily' if i == 0 else 'five_min'

            df[['macd', 'signal_line', 'macd_hist']] = self.engine.calculate_macd(data, [self.fast, self.slow, self.signal], name)
            df.dropna(inplace=True)

            df['signal'] = sg.macd(df['macd_hist'], df['macd'], df['adj_close'], self.signal_type, 
                                   self.combine, self.vote_threshold, self.weights)

            df.rename(columns=dict(log_rets='returns'), inplace=True)
            df['strategy'] = df['returns'] * df['signal']

            if i == 0:
                self.daily = df
            else:
                self.five_min = df

    @property
    def fast(self):
        return self.__fast

    @fast.setter
    def fast(self, value):
        self.__fast = value
        self.__get_data()

    @property
    def slow(self):
        return self.__slow

    @slow.setter
    def slow(self, value):
        self.__slow= value
        self.__get_data()

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, value):
        self.__signal = value
        self.__get_data()

    @property
    def combine(self):
        return self.__combine

    @combine.setter
    def combine(self, value):
        self.__combine = value
        self.__get_data()

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = np.array(value)
        self.__weights /= np.sum(self.__weights)
        self.__get_data()

    @property
    def vote_threshold(self):
        return self.__vote_threshold

    @vote_threshold.setter
    def vote_threshold(self, value):
        self.__vote_threshold = value
        self.__get_data()

    def change_params(self, fast=None, slow=None, signal=None, combine=None, weights=None, vote_threshold=None):
        self.__fast = fast if fast is not None else self.fast
        self.__slow = slow if slow is not None else self.slow
        self.__signal = signal if signal is not None else self.signal
        self.__combine = combine if combine is not None else self.combine
        self.__weights = np.array(weights) if weights is not None else self.weights
        self.__weights /= np.sum(self.__weights)
        self.__vote_threshold = vote_threshold if vote_threshold is not None else self.vote_threshold
        self.__get_data()

    def plot(self, timeframe='1d', start_date=None, end_date=None,
            candlestick=True, show_signal=True):

        df = self.daily if timeframe == '1d' else self.five_min

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        df.dropna(inplace=True)

        fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        specs=[[{"secondary_y": True}],
                        [{"secondary_y": False}]])

        if candlestick:
            price = go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f'{self.asset.ticker} OHLC',
            )
        else:
            price = go.Scatter(
                        x=df.index,
                        y=df['adj_close'],
                        line=dict(
                            color='#2962FF',
                            width=2,
                            dash='solid'
                        ),
                        name=f'{self.asset.ticker} Price',
            )

        MACD = go.Scatter(
                x=df.index,
                y=df['macd'],
                line=dict(color='red', width=1.5),
                name='MACD'
        )

        signal_line = go.Scatter(
                x=df.index,
                y=df['signal_line'],
                line=dict(color='blue', width=1.5),
                name='Signal Line'
        )

        macd_hist = go.Bar(
                x=df.index,
                y=df['macd_hist'],
                marker_color='black',
                name='MACD Histogram'
        )

        if show_signal:
            signal = go.Scatter(
                        x=df.index,
                        y=df['signal'],
                        line=dict(color='green', width=0.8, dash='solid'),
                        name='Buy/Sell signal',
                        yaxis='y2'
            )

        fig.add_trace(price)

        fig.add_trace(MACD, row=2, col=1)
        fig.add_trace(signal_line, row=2, col=1)
        fig.add_trace(macd_hist, row=2, col=1)



        if show_signal:
            fig.add_trace(signal)

        layout = {}

        layout['title'] = dict(
                text=f'{self.asset.ticker} MACD Strategy {self.params}',
                x=0.5,
                y=0.95
            )

        layout['xaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=None,
            )

        layout['yaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=f'Price ({self.asset.currency})',
            )

        layout[f'xaxis1_rangeslider_visible'] = False

        layout['height'] = 800

        if show_signal:
            layout['yaxis2'] = dict(
                    title='Signal',
                    overlaying='y',
                    side='right',
                    range=[-1.1, 1.1],  # Give some padding to the 0/1 signal
                    tickmode='array',
                    tickvals=[-1, 1],
                    ticktext=['Sell', 'Buy']
                )

        layout['yaxis3'] = dict(
                title='MACD',
                side='right'
            )

        layout['legend'] = dict(
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                orientation="h",  # horizontal layout
                bgcolor='rgba(255,255,255,0.8)'
            )

        fig.update_layout(**layout,
                            paper_bgcolor='white',
                            plot_bgcolor='rgba(240,240,240,0.95)',
                            hovermode='x unified')

        fig.show()

        return fig

    def optimize(self, inplace=False, timeframe='1d', start_date=None, end_date=None,
                 slow_range=None, fast_range=None, signal_range=None):

        if fast_range is None:
            fast_range = np.arange(8, 21, 2)
        if slow_range is None:
            slow_range = np.arange(21, 35, 2)
        if signal_range is None:
            signal_range = np.arange(5, 15, 2)

        old_params = {'fast': self.fast, 'slow': self.slow, 'signal': self.signal}

        results = []
        for fast, slow, signal in product(fast_range, slow_range, signal_range):
            if fast >= slow:
                continue

            self.change_params(fast=fast, slow=slow, signal=signal)
            backtest_results = self.backtest(plot=False, 
                             timeframe=timeframe, 
                             start_date=start_date,
                             end_date=end_date)
            results.append((fast, slow, signal, backtest_results['returns'], backtest_results['strategy']))

        results = pd.DataFrame(results, columns=['fast', 'slow', 'signal', 'hold_returns', 'strategy_returns'])
        results['net'] = results['strategy_returns'] - results['hold_returns']
        results = results.sort_values(by='net', ascending=False)

        opt_fast = results.iloc[0]['fast']
        opt_slow = results.iloc[0]['slow']
        opt_signal = results.iloc[0]['signal']

        if inplace:
            self.change_params(fast=opt_fast, slow=opt_slow, signal=opt_signal)
        else:
            self.change_params(**old_params)

        return results


class BB(Strategy):

    def __init__(self, asset, window=20, num_std=2, signal_type=None, combine='weighted', weights=None, vote_threshold=0.5):
        super().__init__(asset)
        self.__window = window
        self.__num_std = num_std

        if signal_type is not None:
            self.signal_type = list(signal_type)
        else:
            self.signal_type = ['bounce', 'double', 'walks', 'squeeze', 'breakout', '%B']

        self.__combine = str(combine)

        if weights is not None:
            self.__weights = np.array(weights)
        else:
            self.__weights = np.array([1 / len(self.signal_type)] * len(self.signal_type))
        self.__weights /= np.sum(self.__weights)

        self.__vote_threshold = vote_threshold
        self.engine = TAEngine()
        self.__get_data()

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])
        self.five_min = pd.DataFrame(self.asset.five_minute[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])
        self.params = f'window={self.window}(±{self.num_std})'

        for i, df in enumerate([self.daily, self.five_min]):
            data = df['adj_close']
            name = 'daily' if i == 0 else 'five_min'

            df[['sma', 'bol_up', 'bol_down']] = self.engine.calculate_bb(data, self.window, self.num_std, name)
            df.dropna(inplace=True)

            df['signal'] = sg.bb(df['adj_close'], df['bol_up'], df['bol_down'], self.signal_type,
                                 self.combine, self.vote_threshold, self.weights)
            df.rename(columns=dict(log_rets='returns'), inplace=True)
            df['strategy'] = df['returns'] * df['signal']
            if i == 0:
                self.daily = df
            else:
                self.five_min = df

    @property
    def window(self):
        return self.__window

    @window.setter
    def window(self, value):
        self.__window = value
        self.__get_data()

    @property
    def num_std(self):
        return self.__num_std

    @num_std.setter
    def num_std(self, value):
        self.__num_std = value
        self.__get_data()

    @property
    def combine(self):
        return self.__combine

    @combine.setter
    def combine(self, value):
        self.__combine = value
        self.__get_data()

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = np.array(value)
        self.__weights /= np.sum(self.__weights)
        self.__get_data()

    @property
    def vote_threshold(self):
        return self.__vote_threshold

    @vote_threshold.setter
    def vote_threshold(self, value):
        self.__vote_threshold = value
        self.__get_data()

    def change_params(self, window=None, num_std=None, combine=None, weights=None, vote_threshold=None):
        self.__window = window if window is not None else self.window
        self.__num_std = num_std if num_std is not None else self.num_std
        self.__combine = combine if combine is not None else self.combine
        self.__weights = np.array(weights) if weights is not None else self.weights
        self.__weights /= np.sum(self.__weights)
        self.__vote_threshold = vote_threshold if vote_threshold is not None else self.vote_threshold
        self.__get_data()

    def plot(self, timeframe='1d', start_date=None, end_date=None,
            candlestick=True, show_signal=True):
        
        df = self.daily if timeframe == '1d' else self.five_min

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        df.dropna(inplace=True)

        fig = go.Figure()

        traces = []

        if candlestick:
            price = go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f'{self.asset.ticker} OHLC',
            )

            sma = go.Scatter(
                        x=df.index,
                        y=df['sma'],
                        line=dict(
                            color='#2962FF',
                            width=2,
                            dash='solid'
                        ),
                        name=f'{self.asset.ticker} SMA',
            )

            traces.extend([price, sma])
        else:
            price = go.Scatter(
                        x=df.index,
                        y=df['adj_close'],
                        line=dict(
                            color='#2962FF',
                            width=2,
                            dash='solid'
                        ),
                        name=f'{self.asset.ticker} Price',
            )

            traces.append(price)

        bol_down = go.Scatter(
            x=df.index,
            y=df['bol_down'],
            line=dict(color='#FF4081', width=1, dash='dash'),
            showlegend=False,
            name='lower band'
        )

        bol_up = go.Scatter(
            x=df.index,
            y=df['bol_up'],
            fill='tonexty',
            line=dict(color='#FF4081', width=1, dash='dash'),
            fillcolor='rgba(68, 68, 255, 0.1)',
            showlegend=False,
            name='upper band'
        )

        traces.extend([bol_down, bol_up])

        if show_signal:
            signal = go.Scatter(
                        x=df.index,
                        y=df['signal'],
                        line=dict(color='green', width=0.8, dash='solid'),
                        name='Buy/Sell signal',
                        yaxis='y2'
            )

            traces.append(signal)

        fig.add_traces(traces)

        layout = {}

        layout['title'] = dict(
                text=f'{self.asset.ticker} BB Strategy {self.params}',
                x=0.5,
                y=0.95
            )

        layout['xaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=None,
            )

        layout['yaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=f'Price ({self.asset.currency})',
            )

        layout[f'xaxis1_rangeslider_visible'] = False

        layout['height'] = 800

        if show_signal:
            layout['yaxis2'] = dict(
                    title='Signal',
                    overlaying='y',
                    side='right',
                    range=[-1.1, 1.1],
                    tickmode='array',
                    tickvals=[-1, 1],
                    ticktext=['Sell', 'Buy']
                )

        layout['legend'] = dict(
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                orientation="h",  # horizontal layout
                bgcolor='rgba(255,255,255,0.8)'
            )

        fig.update_layout(**layout,
                            paper_bgcolor='white',
                            plot_bgcolor='rgba(240,240,240,0.95)',
                            hovermode='x unified')

        fig.show()

        return fig


    def optimize(self, inplace=False, timeframe='1d', start_date=None, end_date=None,
                 window_range=None, num_std_range=None):

        if window_range is None:
            window_range = np.arange(10, 51, 5)
        if num_std_range is None:
            num_std_range = np.arange(1.5, 2.6, 0.1)

        old_params = {'window': self.window, 'num_std': self.num_std}

        results = []
        for window, num_std in product(window_range, num_std_range):

            self.change_params(window=window, num_std=num_std)
            backtest_results = self.backtest(plot=False, 
                             timeframe=timeframe, 
                             start_date=start_date,
                             end_date=end_date)
            results.append((window, num_std, backtest_results['returns'], backtest_results['strategy']))

        results = pd.DataFrame(results, columns=['window', 'num_std', 'hold_returns', 'strategy_returns'])
        results['net'] = results['strategy_returns'] - results['hold_returns']
        results = results.sort_values(by='net', ascending=False)

        opt_window = int(results.iloc[0]['window'])
        opt_num_std = results.iloc[0]['num_std']

        if inplace:
            self.change_params(window=opt_window, num_std=opt_num_std)
        else:
            self.change_params(**old_params)

        return results


class CombinedStrategy(Strategy):

    def __init__(self, asset, strategies=None, combine='weighted', weights=None, vote_threshold=0.5):
        super().__init__(asset)

        if strategies is not None:
            self.__strategies = strategies
        else:
            self.__strategies = [MA_Crossover(asset), RSI(asset), MACD(asset), BB(asset)]

        if weights is not None:
            self.__weights = np.array(weights)
        else:
            self.__weights = np.array([1 / len(self.__strategies)] * len(self.__strategies))
        self.__weights /= np.sum(self.__weights)

        self.__combine = str(combine)
        self.__vote_threshold = vote_threshold

        self.__get_data()

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])
        self.five_min = pd.DataFrame(self.asset.five_minute[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])
        self.params = ''

        for i, df in enumerate([self.daily, self.five_min]):
            name = 'daily' if i == 0 else 'five_min'
            signals = pd.DataFrame(index=df.index)

            for i, strat in enumerate(self.strategies):
                signals[f'{strat.__class__.__name__}_signal_{i}'] = eval(f"strat.{name}['signal']")

            signals.dropna(inplace=True)
            df['signal'] = sg.vote(signals, self.vote_threshold, self.weights)
            df.dropna(inplace=True)

            df.rename(columns=dict(log_rets='returns'), inplace=True)
            df['strategy'] = df['returns'] * df['signal']
            if i == 0:
                self.daily = df
            else:
                self.five_min = df

    @property
    def strategies(self):
        return self.__strategies

    @strategies.setter
    def strategies(self, value):
        self.__strategies = list(value)
        self.__get_data()

    @property
    def combine(self):
        return self.__combine

    @combine.setter
    def combine(self, value):
        self.__combine = value
        self.__get_data()

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = np.array(value)
        self.__weights /= np.sum(self.__weights)
        self.__get_data()

    @property
    def vote_threshold(self):
        return self.__vote_threshold

    @vote_threshold.setter
    def vote_threshold(self, value):
        self.__vote_threshold = value
        self.__get_data()

    def change_params(self, strategies=None, combine=None, weights=None, vote_threshold=None):
        self.__strategies = list(strategies) if strategies is not None else self.strategies
        self.__combine = combine if combine is not None else self.combine
        self.__weights = np.array(weights) if weights is not None else self.weights
        self.__weights /= np.sum(self.__weights)
        self.__vote_threshold = vote_threshold if vote_threshold is not None else self.vote_threshold
        self.__get_data()

    def plot(self, timeframe='1d', start_date=None, end_date=None,
             candlestick=True):
        df = self.daily if timeframe == '1d' else self.five_min

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        df.dropna(inplace=True)

        fig = go.Figure()

        if candlestick:
            price = go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f'{self.asset.ticker} OHLC',
            )
        else:
            price = go.Scatter(
                        x=df.index,
                        y=df['adj_close'],
                        line=dict(
                            color='#2962FF',
                            width=2,
                            dash='solid'
                        ),
                        name=f'{self.asset.ticker} Price',
            )

        signal = go.Scatter(
                    x=df.index,
                    y=df['signal'],
                    line=dict(color='green', width=0.8, dash='solid'),
                    name='Buy/Sell signal',
                    yaxis='y2'
        )

        fig.add_traces([price, signal])

        layout = {}

        layout['title'] = dict(
                text=f'{self.asset.ticker} Combined Strategy {self.params}',
                x=0.5,
                y=0.95
            )

        layout['xaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=None,
            )

        layout['yaxis'] = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title=f'Price ({self.asset.currency})',
            )

        layout[f'xaxis1_rangeslider_visible'] = False

        layout['height'] = 800

        layout['yaxis2'] = dict(
                title='Signal',
                overlaying='y',
                side='right',
                range=[-1.1, 1.1],
                tickmode='array',
                tickvals=[-1, 1],
                ticktext=['Sell', 'Buy']
            )

        layout['legend'] = dict(
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                orientation="h",  # horizontal layout
                bgcolor='rgba(255,255,255,0.8)'
            )

        fig.update_layout(**layout,
                            paper_bgcolor='white',
                            plot_bgcolor='rgba(240,240,240,0.95)',
                            hovermode='x unified')

        fig.show()

        return fig

    def optimize(self, inplace=False, timeframe='1d', start_date=None,
                 end_date=None, threshold_range=None, runs=10):
        return self.optimize_weights(inplace, timeframe, start_date,
                                     end_date, threshold_range, runs)



# TODO:
# parrallelize backtest and optimize methods
# add MACD, ATR and BB strategies
# add transaction costs
# add risk management
# add algo to reduce number of trades (e.g. minimum holding period, dead zone, trend filter)
# add more backtest metrics (e.g. Sharpe ratio, drawdown, max drawdown, win/loss ratio, etc.)
# add more optimization methods (e.g. genetic algorithm, particle swarm optimization, bayesian, walk-forward, rolling)
# add documentation (ugh)
