from abc import ABC, abstractmethod
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Strategy(ABC):

    def __init__(self, asset):
        self.asset = asset

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def backtest(self):
        pass

    @abstractmethod
    def optimize(self):
        pass


class MA_Crossover(Strategy):

    def __init__(self, asset, param_type='window', short_window=20, long_window=50,
                short_alpha=None, long_alpha=None, short_halflife=None, long_halflife=None, ewm=False):

        super().__init__(asset)
        self.ptype = param_type
        self.ewm = ewm
        self.__short = eval(f'short_{param_type}')
        self.__long = eval(f'long_{param_type}')
        self.__get_data()

    def __str__(self):
        return f'MA_Crossover({self.asset.ticker}, short_{self.ptype}={self.short}, long_{self.ptype}={self.long})'

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily[['adj_close', 'log_rets']])
        self.five_min = pd.DataFrame(self.asset.five_minute[['adj_close', 'log_rets']])

        for i, timeframe in enumerate([self.daily, self.five_min]):
            df = timeframe
            if self.ewm:
                param = {}
                key = self.ptype if self.ptype != 'window' else 'span'
                param[key] = self.short
                df['short'] = df['adj_close'].ewm(**param).mean()
                param[key] = self.long
                df['long'] = df['adj_close'].ewm(**param).mean()
            else:
                df['short'] = df['adj_close'].rolling(window=self.short).mean()
                df['long'] = df['adj_close'].rolling(window=self.long).mean()
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
        if ewm is not None:
            self.ewm = ewm
        self.__get_data()

    def plot(self, timeframe='1d', start_date=None, end_date=None, 
            show_signal=True, fig=None, subplot_idx=None):

        df = self.daily if timeframe == '1d' else self.five_min

        long_data = df['long']
        short_data = df['short']

        if start_date is not None:
            long_data = long_data[long_data.index >= start_date]
            short_data = short_data[short_data.index >= start_date]
        if end_date is not None:
            long_data = long_data[long_data.index <= end_date]
            short_data = short_data[short_data.index <= end_date]

        long_data = long_data.dropna()
        short_data = short_data.dropna()

        common_index = long_data.index.intersection(short_data.index)
        long_data = long_data.reindex(common_index)
        short_data = short_data.reindex(common_index)

        if show_signal:
            signal = df['signal'].reindex(common_index)

        short_param = f'{self.ptype}={self.short}'
        long_param = f'{self.ptype}={self.long}'

        # Add short MA line
        trace1 = go.Scatter(
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
        trace2 = go.Scatter(
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
            trace3 = go.Scatter(
                x=signal.index,
                y=signal,
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

        if standalone:
            fig.show()
        else:
            fig.update_yaxes(
                title_text=f'Price ({self.asset.currency})',
                row=subplot_idx[0] if subplot_idx else None, 
                col=subplot_idx[1] if subplot_idx else None
            )
            fig.update_xaxes(
                title_text=f'{self.asset.ticker} MA Crossover ({self.short}/{self.long})', 
                row=subplot_idx[0] if subplot_idx else None, 
                col=subplot_idx[1] if subplot_idx else None
            )

        return fig

    def backtest(self, plot=True, timeframe='1d', start_date=None, end_date=None, 
            show_signal=True, fig=None, subplot_idx=None):

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

            # Add long MA line
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
                        text=f'{self.asset.ticker} MA Crossover Backtest ({self.short}/{self.long})',
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
                    title_text=f'{self.asset.ticker} MA Crossover Backtest ({self.short}/{self.long})', 
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )

        return np.exp(df[['returns', 'strategy']].sum())

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
            self.short = opt_short
            self.long = opt_long
        else:
            self.change_params(**old_params)

        return results


class RSI(Strategy):

    def __init__(self, asset, ub=70, lb=30, window=14, switch='re', m_rev=True):

        super().__init__(asset)
        self.ub = ub
        self.lb = lb
        self.window = window
        self.switch = switch
        self.m_rev = m_rev
        self.__get_data()

    def __get_data(self):
        self.daily = self.asset.daily[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']]
        self.five_min = self.asset.five_minute[['open', 'high', 'low', 'close', 'adj_close', 'log_rets']]

        for i, timeframe in enumerate([self.daily, self.five_min]):
            df = timeframe
            delta = df['adj_close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/self.window, min_periods=self.window).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.window, min_periods=self.window).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df.dropna(inplace=True)

            if self.switch == 're':
                df['signal'] = np.where(
                    np.logical_and(df['rsi'].shift(1) > self.ub, df['rsi'] < ub), -1,
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

            df.rename(columns=dict(log_rets='returns'), inplace=True)
            df['strategy'] = df['returns'] * df['signal']

            if i == 0:
                self.daily = df
            else:
                self.five_min = df

    def backtest(self):
        pass

    def optimize(self):
        pass

    def plot(self):
        pass

# TODO:
# parrallelize backtest and optimize methods
# add RSI and MACD strategies
# add transaction costs
# add risk management
# add algo to reduce number of trades (e.g. minimum holding period, dead zone, trend filter)
# add more backtest metrics (e.g. Sharpe ratio, drawdown, max drawdown, win/loss ratio, etc.)
# add more optimization methods (e.g. genetic algorithm, particle swarm optimization, bayesian, walk-forward, rolling)
# add documentation (ugh)
