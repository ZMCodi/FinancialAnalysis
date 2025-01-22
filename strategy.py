from abc import ABC, abstractmethod
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt

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

    def __get_data(self):
        self.daily = pd.DataFrame(self.asset.daily['adj_close'])
        self.five_min = pd.DataFrame(self.asset.five_minute['adj_close'])

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

    def change_params(self, param_type, short, long, ewm):
        self.ptype = param_type
        self.short = short
        self.long = long
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
            signal = (short_data > long_data).astype(int)

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
                    range=[-0.1, 1.1],  # Give some padding to the 0/1 signal
                    tickmode='array',
                    tickvals=[0, 1],
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
                title_text=f'Price ({self.currency})',
                row=subplot_idx[0] if subplot_idx else None, 
                col=subplot_idx[1] if subplot_idx else None
            )
            fig.update_xaxes(
                title_text=f'{self.ticker} MA Crossover ({self.short}/{self.long})', 
                row=subplot_idx[0] if subplot_idx else None, 
                col=subplot_idx[1] if subplot_idx else None
            )

        return fig

    def backtest(self):
        pass

    def optimize(self):
        pass

# TODO:
# implement the other methods