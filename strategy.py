from abc import ABC, abstractmethod
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class TAEngine:

    def __init__(self):
        self.cache = {}

    def calculate_ma(self, data, ewm, param_type, param, name):
        key = f'ma_{param_type}={param}, {name=}'
        if key in self.cache:
            return self.cache[key]

        if ewm:
            self.cache[key] = data.ewm(**{f'{param_type}': param}).mean()
        else:
            self.cache[key] = data.rolling(window=param).mean()

        return self.cache[key]

    def calculate_rsi(self, data, window, name):
        key = f'rsi_window={window}, {name=}'
        if key in self.cache:
            return self.cache[key]

        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
        rs = gain / loss
        self.cache[key] = 100 - (100 / (1 + rs))

        return self.cache[key]

class Strategy(ABC):

    def __init__(self, asset):
        self.asset = asset

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    def backtest(self, plot=True, timeframe='1d', start_date=None, end_date=None, 
            show_signal=True, fig=None, subplot_idx=None):

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
                        text=f'{self.asset.ticker} {name} Backtest ({self.p1}/{self.p2})',
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
                    title_text=f'{self.asset.ticker} {name} Backtest ({self.p1}/{self.p2})', 
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )

        return np.exp(df[['returns', 'strategy']].sum())


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
        self.p1 = self.short
        self.p2 = self.long

        for i, df in enumerate([self.daily, self.five_min]):
            data = df['adj_close']
            name = 'daily' if i == 0 else 'five_min'
            ptype = 'span' if self.ptype == 'window' and self.ewm else self.ptype

            df['short'] = self.engine.calculate_ma(data, self.ewm, ptype, self.short, name)
            df['long'] = self.engine.calculate_ma(data, self.ewm, ptype, self.long, name)

            df['signal'] = np.where(df['short'] > df['long'], 1, -1)
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

    def change_params(self, param_type, short, long, ewm=None):
        self.ptype = param_type
        self.short = short
        self.long = long
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
                text=f'{self.asset.ticker} MA Crossover ({self.short}/{self.long})',
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
                short_range = np.arange(0.05, 0.31, 0.05)  # alpha
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
            self.change_params(self.ptype, short, long)
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
            self.change_params(self.ptype, opt_short, opt_long)
        else:
            self.change_params(**old_params)

        return results


class RSI(Strategy):

    def __init__(self, asset, ub=70, lb=30, window=14, exit='re', m_rev=True, m_rev_bound=50):

        super().__init__(asset)
        self.__ub = ub
        self.__lb = lb
        self.window = window
        self.exit = exit
        self.m_rev = m_rev
        self.__m_rev_bound = m_rev_bound
        self.engine = TAEngine()
        self.__get_data()

    def __str__(self):
        return f'RSI({self.asset.ticker}, ub={self.ub}, lb={self.lb}, window={self.window}, exit={self.exit}, m_rev={self.m_rev}, m_rev_bound={self.m_rev_bound})'

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])
        self.five_min = pd.DataFrame(self.asset.five_minute[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']])

        self.p1 = self.ub
        self.p2 = self.lb

        for i, df in enumerate([self.daily, self.five_min]):
            data = df['adj_close']
            name = 'daily' if i == 0 else 'five_min'

            df['rsi'] = self.engine.calculate_rsi(data, self.window, name)
            df.dropna(inplace=True)

            if self.exit == 're':
                df['signal'] = np.where(
                    np.logical_and(df['rsi'].shift(1) > self.ub, df['rsi'] < self.ub), -1,
                    np.where(np.logical_and(df['rsi'].shift(1) < self.lb, df['rsi'] > self.lb), 1, np.nan)
                )
            else:
                df['signal'] = np.where(
                    df['rsi'] > self.ub, -1,
                    np.where(df['rsi'] < self.lb, 1, np.nan)
                )

            if np.isnan(df['signal'].iloc[0]):
                idx = df.index[0]
                df.loc[idx, 'signal'] = 1
            df['signal'] = df['signal'].ffill().astype(int)

            if self.m_rev:
                if self.exit == 're':
                    short_entries = np.logical_and(df['rsi'].shift(1) > self.ub, df['rsi'] < self.ub)
                else:
                    short_entries = np.logical_and(df['rsi'].shift(1) <= self.ub, df['rsi'] > self.ub)

                mean_rev_points = np.logical_and(df['rsi'] <= self.m_rev_bound, df['signal'] == -1)
                groups = short_entries.cumsum()
                mean_rev_triggered = mean_rev_points.groupby(groups).cummax()
                df['signal'] = np.where(mean_rev_triggered, 1, df['signal'])

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
    def m_rev_bound(self):
        return self.__m_rev_bound

    @m_rev_bound.setter
    def m_rev_bound(self, value):
        self.__m_rev_bound = value
        self.__get_data()

    def change_params(self, ub, lb, window=None, exit=None, m_rev=None, m_rev_bound=None):
        self.ub = ub
        self.lb = lb
        self.window = window if window is not None else self.window
        self.exit = exit if exit is not None else self.exit
        self.m_rev = m_rev if m_rev is not None else self.m_rev
        self.m_rev_bound = m_rev_bound if m_rev_bound is not None else self.m_rev_bound
        self.__get_data()

    def plot(self, timeframe='1d', start_date=None, end_date=None, rsi=True,
            candlestick=True, show_signal=True):

        df = self.daily if timeframe == '1d' else self.five_min

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        df.dropna(inplace=True)

        if rsi:
            fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.7, 0.3],
                            specs=[[{"secondary_y": True}],
                            [{"secondary_y": False}]])
        else:
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
                        y=df['close'],
                        line=dict(
                            color='#2962FF',
                            width=2,
                            dash='solid'
                        ),
                        name=f'{self.asset.ticker} Price',
            )

        if rsi:
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

        fig.add_trace(price, row=1 if rsi else None, col=1 if rsi else None)

        if rsi:
            fig.add_trace(RSI, row=2, col=1)
            fig.add_hline(y=self.ub, row=2, col=1)
            fig.add_hline(y=self.lb, row=2, col=1)

        if show_signal:
            fig.add_trace(signal,
                        row=1 if rsi else None, 
                        col=1 if rsi else None, 
                        secondary_y=True if rsi else None)

        layout = {}

        layout['title'] = dict(
                text=f'{self.asset.ticker} RSI Strategy ({self.ub}/{self.lb})',
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

        if rsi:
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
                    'exit': self.exit, 'm_rev': self.m_rev, 'm_rev_bound': self.m_rev_bound}

        results = []
        for ub, lb, window, m_rev_bound in product(*params):
            self.change_params(ub, lb, window, m_rev_bound=m_rev_bound)
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
            self.change_params(opt_ub, opt_lb, opt_window, m_rev_bound=opt_m_rev_bound)
        else:
            self.change_params(**old_params)

        return results


# TODO:
# parrallelize backtest and optimize methods
# add MACD and BB strategies
# add transaction costs
# add risk management
# add algo to reduce number of trades (e.g. minimum holding period, dead zone, trend filter)
# add more backtest metrics (e.g. Sharpe ratio, drawdown, max drawdown, win/loss ratio, etc.)
# add more optimization methods (e.g. genetic algorithm, particle swarm optimization, bayesian, walk-forward, rolling)
# add documentation (ugh)
