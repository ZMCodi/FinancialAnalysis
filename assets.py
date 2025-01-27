''' Asset module which makes data querying, analysis and plotting simple
Just pass in a ticker from yfinance and analyze the asset in just a few lines of code
- statistical analysis
- resample
- price history plot
- candlestick with volume plot
- MA plot
- returns distribution plot
'''

import yfinance as yf
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import psycopg as pg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import plotly.express as px
from datetime import datetime, date
import kaleido
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from config import DB_CONFIG
from typing import Self, Optional

DateLike = str | datetime | date | pd.Timestamp

class Asset():
    ''' Asset class handles all the data processing and plotting functions
    - Queries data from the database
    - if not available, download from yfinance and add to db to set up future tracking
    - prepares data for analysis and plotting
    '''

    def __init__(self, ticker: str) -> None:
        '''Instantiates the asset class and gets data from the database

        Args:
            ticker (str): ticker string from yfinance
        '''
        self.ticker = ticker
        self.__get_data()

    def __repr__(self) -> str:
        return f'Asset({self.ticker!r})'
    
    def __eq__(self, other: Self) -> bool:
        return self.ticker == other.ticker
    
    def __hash__(self) -> int:
        return hash(self.ticker)

    def __get_data(self) -> None:
        """Gets data from the database and calculate additional columns

        - Tries to query db for ticker data
        - If not available, add new data
        - Calculate returns and log returns
        - Store ticker metadata
        """
        # query db and check if ticker exists
        with pg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT date, open, high, low, close, adj_close, volume FROM daily WHERE ticker = %s", (self.ticker,))
                # if not, add new ticker and requery
                if cur.rowcount == 0:
                    self.__insert_new_ticker()
                    cur.execute("SELECT date, open, high, low, close, adj_close, volume FROM daily WHERE ticker = %s", (self.ticker,))

                data = cur.fetchall()

                # reindex data and calculate returns and log returns
                self.daily = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']).set_index('date')
                self.daily = self.daily.astype(float)
                self.daily.index = pd.to_datetime(self.daily.index)
                self.daily = self.daily.sort_index()
                self.daily['log_rets'] = np.log(self.daily['adj_close'] / self.daily['adj_close'].shift(1))
                self.daily['rets'] = self.daily['adj_close'].pct_change()

                # same for five minute data
                cur.execute("SELECT date, open, high, low, close, adj_close, volume FROM five_minute WHERE ticker = %s", (self.ticker,))
                self.five_minute = pd.DataFrame(cur.fetchall(), columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']).set_index('date')
                self.five_minute = self.five_minute.astype(float)
                self.five_minute.index = pd.to_datetime(self.five_minute.index)
                self.five_minute = self.five_minute.sort_index()
                self.five_minute['log_rets'] = np.log(self.five_minute['adj_close'] / self.five_minute['adj_close'].shift(1))
                self.five_minute['rets'] = self.five_minute['adj_close'].pct_change()

                # get metadata like asset type and currency
                cur.execute("SELECT asset_type FROM tickers WHERE ticker = %s", (self.ticker,))
                self.asset_type = cur.fetchone()[0]

                cur.execute("SELECT currency FROM tickers WHERE ticker = %s", (self.ticker,))
                self.currency = cur.fetchone()[0]
    
    def __insert_new_ticker(self) -> None:
        """Downloads and inserts new ticker to database
        
        - Downloads ticker from yfinance
        - Get ticker metadata and insert into tickers table
        - Clean and transform daily and 5min data
        - Batch insert into database
        """
        print(f"{self.ticker} is not yet available in database. Downloading from yfinance...")

        # Ticker table data
        ticker = yf.Ticker(self.ticker)

        # Check valid ticker
        if ticker.history().empty:
            print(f'{self.ticker} is an invalid yfinance ticker')
            return

        # transform metadata for insertion to database
        comp_name = ticker.info['shortName'].replace("'", "''")
        exchange = ticker.info['exchange']
        currency = ticker.info['currency'].upper()
        start_date = pd.to_datetime('today').date()
        asset_type = ticker.info['quoteType'].lower()

        # map exchanges according to pandas market calendar
        exchange_mapping = {
            'NYQ': 'NYSE',
            'NMS': 'NASDAQ',
            'NGM': 'NASDAQ'
        }
        exchange = exchange_mapping.get(exchange, exchange)

        # special handling for optional metadata
        try:
            market_cap = ticker.info['marketCap']
        except KeyError:
            market_cap = None

        try:
            sector = ticker.info['sector']
        except KeyError:
            sector = None

        print(f'Inserting to DB {ticker=}, {comp_name=}, {exchange=}, {currency=}, {asset_type=}, {market_cap=}, {sector=}')

        # Get daily data from 2020, clean and transform
        daily_data = yf.download(self.ticker, start='2020-01-01')
        daily_data = daily_data.droplevel(1, axis=1)
        daily_data['ticker'] = self.ticker
        clean_daily = self.__clean_data(daily_data)
        clean_daily = clean_daily.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        # adj_close column might not be available, especially for crypto
        if 'Adj Close' not in clean_daily.columns:
            clean_daily['adj_close'] = clean_daily['close']
        else:
            clean_daily = clean_daily.rename(columns={'Adj Close': 'adj_close'})

        # Get 5min data
        five_min_data = yf.download(self.ticker, interval='5m')
        five_min_data = five_min_data.droplevel(1, axis=1)
        five_min_data['ticker'] = self.ticker
        clean_five_min = self.__clean_data(five_min_data)
        clean_five_min = clean_five_min.rename(columns={'Datetime': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        if 'Adj Close' not in clean_five_min.columns:
            clean_five_min['adj_close'] = clean_five_min['close']
        else:
            clean_five_min = clean_five_min.rename(columns={'Adj Close': 'adj_close'})

        # Insert to database
        with pg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT DISTINCT LEFT(currency_pair, 3) FROM daily_forex;')
                currencies = [cur[0] for cur in cur.fetchall()]

                # if asset's currency is not in db, add all possible pairs
                if currency not in currencies:
                    self.__add_new_currency(cur, conn, currencies, currency)

                # insert metadata into db
                cur.execute("INSERT INTO tickers (ticker, comp_name, exchange, sector, market_cap, start_date, currency, asset_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (self.ticker, comp_name, exchange, sector, market_cap, start_date, currency, asset_type))

                # batch insertion for price data
                BATCH_SIZE = 1000
                batch_count = 0
                rows_inserted = 0

                # insert daily data
                for _, row in clean_daily.iterrows():
                    cur.execute("INSERT INTO daily (ticker, date, open, high, low, close, adj_close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (row['ticker'], row['date'], row['open'], row['high'], row['low'], row['close'], row['adj_close'], row['volume']))
                    batch_count += 1
                    rows_inserted += 1

                    if batch_count >= BATCH_SIZE:
                        conn.commit()
                        batch_count = 0

                if batch_count > 0:
                    conn.commit()
                    batch_count = 0

                print(f'daily_{rows_inserted=}')
                rows_inserted = 0

                # insert 5min data
                for _, row in clean_five_min.iterrows():
                    cur.execute("INSERT INTO five_minute (ticker, date, open, high, low, close, adj_close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (row['ticker'], row['date'], row['open'], row['high'], row['low'], row['close'], row['adj_close'], row['volume']))
                    batch_count += 1
                    rows_inserted += 1

                    if batch_count >= BATCH_SIZE:
                        conn.commit()
                        batch_count = 0

                if batch_count > 0:
                    conn.commit()
                    batch_count = 0

                print(f'five_min_{rows_inserted=}')


    def __clean_data(self, df: DataFrame) -> DataFrame:
        """Cleans OHLC data to ensure data passes db table price checks

        Args:
            df (pandas.core.frame.DataFrame): df to be cleaned

        Returns:
            pandas.core.frame.DataFrame: cleaned df
        """
        # ensure appropriate high and low values
        mask = (df['High'] < df['Open']) | (df['High'] < df['Close']) | (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
        clean = df[~mask].copy()
        temp = df[mask].copy()

        temp['High'] = temp[['Open', 'Close', 'High']].max(axis=1)
        temp['Low'] = temp[['Open', 'Close', 'Low']].min(axis=1)
        clean = pd.concat([clean, temp], axis=0)
        clean = clean.reset_index()

        return clean

    def __add_new_currency(self, cur: pg.Cursor, conn: pg.Connection, currencies: list, currency: str) -> None:
        """Adds new currency to database to be tracked

        Args:
            cur (psycopg.Cursor): connection cursor to execute data insertion
            conn (psycopg.Connection): Database connection
            currencies (list): List of currencies in database
            currency (str): New currency to be added
        """
        # create all possible pairs and format according to yfinance
        new_forex = []
        for curr in currencies:
            new_forex.append(f'{curr}{currency}=X')
            new_forex.append(f'{currency}{curr}=X')

        # get new forex data from yfinance
        df_list = []
        for pair in new_forex:
            data = yf.download(pair, start='2020-01-01')
            data = data.droplevel(1, axis=1)
            data['currency_pair'] = f'{pair[:3]}/{pair[3:6]}'  # reformat according to db
            df_list.append(data)

        # combine, clean and transform data
        df = pd.concat(df_list)
        clean = self.clean_data(df)
        clean.drop(columns=['Adj Close', 'Volume'], inplace=True)
        clean = clean.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})

        # batch insertion
        BATCH_SIZE = 1000
        batch_count = 0
        for _, row in clean.iterrows():
            cur.execute("INSERT INTO daily_forex (currency_pair, date, open, high, low, close) VALUES (%s, %s, %s, %s, %s, %s)", (row['currency_pair'], row['date'], row['open'], row['high'], row['low'], row['close']))
            batch_count += 1

            if batch_count >= BATCH_SIZE:
                conn.commit()
                batch_count = 0

        if batch_count > 0:
            conn.commit()

    def plot_price_history(self, *, timeframe: str = '1d', start_date: Optional[DateLike] = None,
                        end_date: Optional[DateLike] = None, resample: Optional[str] = None, 
                        interactive: bool = True, line: Optional[int | float] = None, filename: Optional[str] = None, 
                        fig: Optional[go.Figure | Figure] = None, subplot_idx: Optional[int | tuple[int, int]] = None) -> go.Figure | Figure:
        """Plots the price history of the underlying asset.

        Args:
            timeframe (str, optional): Determines which dataframe to use ('1d' for daily or '5m' for five minute data). 
                Defaults to '1d'.
            start_date (DateLike, optional): Start date for plotting range. Defaults to None.
            end_date (DateLike, optional): End date for plotting range. Defaults to None.
            resample (str, optional): Resampling frequency in pandas format (e.g., 'B' for business day).
                Used primarily with 5-min data to remove flat regions during market close. Defaults to None.
            interactive (bool, optional): Whether to create an interactive Plotly graph (True) 
                or static Matplotlib graph (False). Defaults to True.
            line (float | int, optional): Y-value for horizontal threshold line. Defaults to None.
            filename (str, optional): Path to save the figure. Defaults to None.
            fig (plotly.graph_objects.Figure | matplotlib.figure.Figure, optional): Existing figure to plot on.
                Used for overlaying plots or creating subplots. Defaults to None.
            subplot_idx (int | tuple[int, int], optional): Subplot position specifier.
                For Matplotlib: integer index.
                For Plotly: (row, col) tuple. Defaults to None.

        Returns:
            (plotly.graph_objects.Figure | matplotlib.figure.Figure): Price history of underlying asset.
                Returns Plotly figure if interactive=True, Matplotlib figure otherwise.
        """

        # choose data based on specified timeframe
        data = self.daily['close'] if timeframe == '1d' else self.five_minute['close']

        # slice data according to specified start and end dates
        if start_date is not None:
            data = data[data.index >= start_date]
        if end_date is not None:
            data = data[data.index <= end_date]

        if resample is not None:
            data = data.resample(resample).last()

        data = data.dropna()

        # static plot
        if not interactive:
            # Set the style and figure size
            plt.style.use('seaborn-v0_8')

            # create figure if not passed or get axes from figure
            standalone = True
            if fig is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            else:
                ax = fig.axes[subplot_idx] if subplot_idx is not None else fig.gca()
                standalone = False

            # Create the plot with customizations
            sns.lineplot(data=data.to_frame(), x=data.index, y=data, ax=ax,
                        color='#1f77b4',
                        linewidth=2)

            # Customize the title and labels
            ax.set_title(f'{self.ticker} Price History', 
                        fontsize=16, 
                        pad=20,
                        fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(f'Price ({self.currency})', fontsize=12)

            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

            def format_prices(x, p):
                if x >= 1e3:
                    return f'{x/1e3:.1f}K'
                else:
                    return f'{x:.1f}'

            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_prices))

            # Customize the spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.tick_params(labelsize=10)
            plt.xticks(rotation=0)

            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')

            # add line at specified height
            if line is not None:
                ax.axhline(line, color='r', linestyle='--')

            if filename is not None:
                fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=(True if filename.endswith('.png') else False))

            if standalone:
                plt.show()

        # interactive plot
        else:

            # keeps track of whether plot is a subplot or standalone
            standalone = False
            if fig is None:
                fig = go.Figure()
                standalone = True

            # Add price trace
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data,
                    name=f'{self.ticker} Price',
                    connectgaps=True
                ),
                row=subplot_idx[0] if subplot_idx else None,
                col=subplot_idx[1] if subplot_idx else None
            )

            # add line
            if line is not None:
                fig.add_hline(y=line,
                    line_dash="dash",
                    line_color="red",
                )

            # Only add title and show plot if standalone
            if standalone:
                fig.update_layout(
                    title=f'{self.ticker} Price History',
                    xaxis_title='Date',
                    yaxis_title=f'Price ({self.currency})'
                )

                fig.show()

            # if subplots, only update the axes
            else:
                fig.update_yaxes(
                    title_text=f'Price ({self.currency})', 
                    row=subplot_idx[0] if subplot_idx else None,
                    col=subplot_idx[1] if subplot_idx else None
                )
                fig.update_xaxes(
                    title_text=f'{self.ticker} Price History', 
                    row=subplot_idx[0] if subplot_idx else None,
                    col=subplot_idx[1] if subplot_idx else None
                )

            if filename is not None:
                fig.write_image(filename)

        return fig


    def plot_candlestick(self, *, timeframe: str = '1d', start_date: Optional[DateLike] = None, end_date: Optional[DateLike] = None, 
                         resample: Optional[str] = None, interactive: bool = True, volume: bool = True,
                         filename: Optional[str] = None, fig: Optional[go.Figure | Figure] = None, 
                         candle_idx: Optional[int | tuple[int, int]] = None, vol_idx: Optional[int | tuple[int, int]] = None) -> go.Figure | Figure:
        """Plots the candlestick chart of the underlying asset.

        Args:
            timeframe (str, optional): Determines which dataframe to use ('1d' for daily or '5m' for five minute data). 
                Defaults to '1d'.
            start_date (DateLike, optional): Start date for plotting range. Defaults to None.
            end_date (DateLike, optional): End date for plotting range. Defaults to None.
            resample (str, optional): Resampling frequency in pandas format (e.g., 'B' for business day).
            interactive (bool, optional): Whether to create an interactive Plotly graph (True) 
                or static Matplotlib graph (False). Defaults to True.
            volume (bool): Whether to plot volume bars or not. Defaults to True
            filename (str, optional): Path to save the figure. Defaults to None.
            fig (plotly.graph_objects.Figure | matplotlib.figure.Figure, optional): Existing figure to plot on.
                Used for overlaying plots or creating subplots. Defaults to None.
            candle_idx (int | tuple[int, int], optional): Candlestick subplot position specifier.
                For Matplotlib: integer index.
                For Plotly: (row, col) tuple. Defaults to None.
            vol_idx (int | tuple[int, int], optional): Volume bars subplot position specifier.
                For Matplotlib: integer index.
                For Plotly: (row, col) tuple. Defaults to None.

        Returns:
            (plotly.graph_objects.Figure | matplotlib.figure.Figure): Candlestick chart of underlying asset.
                Returns Plotly figure if interactive=True, Matplotlib figure otherwise.
        """

        if resample is not None:
            data = self.resample(period=resample, five_min=True if timeframe != '1d' else False)
        else:
            data = self.daily if timeframe == '1d' else self.five_minute

        if end_date is not None:
            data = data[data.index <= end_date]
        if start_date is not None:
            data = data[data.index >= start_date]

        # default time window for static plot to ensure plot is readable
        else:
            if not interactive:
                if timeframe == '1d':
                    days_back = 365
                else:
                    if self.asset_type == 'cryptocurrency':
                        days_back = 1
                    else:
                        days_back = 3
                data = data[data.index >= data.index[-1] - pd.Timedelta(days=days_back)]

        data = data.dropna()

        # static plot
        if not interactive:

            plt.style.use('seaborn-v0_8')

            # create figure and axes or use existing if passed
            if fig is None:
                standalone = True
                if volume:
                    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), height_ratios=(3, 1))
                else:
                    fig, ax1 = plt.subplots(figsize=(12, 6))
            else:
                standalone = False
                if volume:
                    ax1 = fig.axes[candle_idx] if candle_idx is not None else fig.gca()
                    ax2 = fig.axes[vol_idx] if vol_idx is not None else fig.axes[candle_idx + 1]
                else:
                    ax1 = fig.axes[candle_idx] if candle_idx is not None else fig.gca()

            fig.patch.set_facecolor('white')
            ax1.set_facecolor('white')

            # define colors and styles
            mc = mpf.make_marketcolors(up='#26A69A',
                                    down='#EF5350', 
                                    edge='inherit',
                                    volume='inherit')

            s = mpf.make_mpf_style(marketcolors=mc,
                                gridstyle=':',
                                gridcolor='#E0E0E0',
                                y_on_right=False)

            # create candlestick plot
            mpf.plot(data,
                    type='candle',
                    style=s,
                    ax=ax1,
                    datetime_format='%Y-%m-%d')

            ax1.grid(True, linestyle=':', color='#E0E0E0', alpha=0.6)

            # Add title
            ax1.set_title(f"{self.ticker} Candlestick Chart{' with Volume Bars' if volume else ''}", 
                        pad=20, 
                        fontsize=14, 
                        fontweight='bold',
                        fontfamily='sans-serif')

            # Refine x-axis formatting
            n_labels = 6 if standalone else 3
            step = len(data) // n_labels

            # Clean up candlestick axis
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.tick_params(labelsize=10)
            ax1.set_ylabel(f'Price ({self.currency})', fontsize=10, fontfamily='sans-serif')

            ax1.set_xticks(range(0, len(data), step))
            ax1.set_xticklabels(data.index[::step].strftime('%Y-%m-%d'), rotation=0)

            def format_prices(x, p):
                if x >= 1e3:
                    return f'{x/1e3:.1f}K'
                else:
                    return f'{x:.1f}'

            ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_prices))

            # add volume bars
            if volume:
                if standalone:
                    ax1.set_xticklabels([])
                ax2.set_facecolor('white')

                # format colors according to candlestick
                colors = ['#26A69A' if close >= open else '#EF5350'
                        for open, close in zip(data['open'], data['close'])]

                ax2.bar(range(len(data)), 
                        data['volume'], 
                        alpha=0.8,
                        color=colors)

                ax2.grid(True, linestyle=':', color='#E0E0E0', alpha=0.6)

                ax2.set_xticks(range(0, len(data), step))
                ax2.set_xticklabels(data.index[::step].strftime('%Y-%m-%d'), rotation=0)

                # Clean up volume axis
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.tick_params(labelsize=10)
                ax2.set_ylabel('Volume', fontsize=10, fontfamily='sans-serif')

                # Link the x-axes
                ax1.set_xlim(ax2.get_xlim())

                def format_volume(x, p):
                    if x >= 1e9:
                        return f'{x/1e9:.1f}B'
                    elif x >= 1e6:
                        return f'{x/1e6:.1f}M'
                    else:
                        return f'{x/1e3:.1f}K'

                ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_volume))

                # Add padding for volume subplot
                plt.subplots_adjust(bottom=0.15)
            else:
                # Format x-axis for single plot
                n_labels = 6 if standalone else 3
                step = len(data) // n_labels
                ax1.set_xticks(range(0, len(data), step))
                ax1.set_xticklabels(data.index[::step].strftime('%Y-%m-%d'), rotation=0)

            if filename is not None:
                fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=(True if filename.endswith('.png') else False))

            if standalone:
                plt.show()

        # interactive plot
        else:

            # create or use existing figure
            standalone = False
            if fig is None:
                standalone = True
                if volume:
                    fig = make_subplots(rows=2, cols=1, 
                                    shared_xaxes=True, 
                                    vertical_spacing=0.03, 
                                    row_heights=[0.7, 0.3])
                else:
                    fig = go.Figure()

            if standalone:
                candlestick_row = 1
                candlestick_col = 1
            else:
                candlestick_row = candle_idx[0] if candle_idx is not None else 1
                candlestick_col = candle_idx[1] if candle_idx is not None else 1

            # Create candlestick trace
            candlestick = go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC'
            )

            # Add trace with appropriate positioning
            if not standalone:  # subplot case (with or without volume)
                fig.add_trace(
                    candlestick,
                    row=candlestick_row,
                    col=candlestick_col
                )
            else:  # standalone case
                fig.add_trace(candlestick)

            # Add volume only if requested
            if volume:
                if vol_idx is not None:
                    vol_row, vol_col = vol_idx
                else:
                    vol_row, vol_col = 2, 1

                colors = ['#26A69A' if close >= open else '#EF5350' 
              for open, close in zip(data['open'], data['close'])]

                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['volume'],
                        name='Volume',
                        marker=dict(
                            color=colors,
                            line=dict(color=colors)
                        )
                    ),
                    row=vol_row,
                    col=vol_col
                )

            # Update layout
            title = f'{self.ticker} Candlestick Chart'
            if volume:
                title += ' with Volume Bars'

            layout_updates = {
                f'xaxis{candlestick_row}_rangeslider_visible': False,
                'height': 800 if volume else 600
            }

            if volume:
                layout_updates[f'xaxis{vol_row}_rangeslider_visible'] = False

            if standalone:
                layout_updates['title'] = title

            # Update y-axes labels
            if standalone:  # Single figure
                layout_updates['yaxis_title'] = f'Price ({self.currency})'
            else:  # Part of subplots
                fig.update_yaxes(title_text=f'Price ({self.currency})', row=candlestick_row, col=candlestick_col)
                fig.update_xaxes(
                    title_text=f'{self.ticker} Candlestick', row=candlestick_row, col=candlestick_col
                )
                if volume:
                    fig.update_yaxes(title_text="Volume", row=vol_row, col=vol_col)
                    fig.update_xaxes(
                        title_text=f'{self.ticker} Volume', row=vol_row, col=vol_col
                    ) 

            fig.update_layout(**layout_updates)

            # Only show if it's a standalone figure
            if standalone:
                fig.show()

            if filename is not None:
                fig.write_image(filename)

        return fig

    def plot_SMA(self, *, window: int = 20, timeframe: str = '1d', start_date: Optional[DateLike] = None, end_date: Optional[DateLike] = None,
                 resample: Optional[str] = None, interactive: bool = True, r: float = 0., ewm: bool = False, 
                 alpha: Optional[float] = None, halflife: Optional[float] = None, bollinger_bands: bool = False, num_std: float = 2.,  
                 filename: Optional[str] = None, fig: Optional[go.Figure | Figure] = None, 
                 subplot_idx: Optional[int | tuple[int, int]] = None) -> go.Figure | Figure:
        """Plots the moving average price of the underlying asset.

        Args:
            window (int): Rolling window. Defaults to 20
            timeframe (str, optional): Determines which dataframe to use ('1d' for daily or '5m' for five minute data). 
                Defaults to '1d'.
            start_date (DateLike, optional): Start date for plotting range. Defaults to None.
            end_date (DateLike, optional): End date for plotting range. Defaults to None.
            resample (str, optional): Resampling frequency in pandas format (e.g., 'B' for business day).
                Used primarily with 5-min data to remove flat regions during market close. Defaults to None.
            interactive (bool, optional): Whether to create an interactive Plotly graph (True) 
                or static Matplotlib graph (False). Defaults to True.
            r (float): risk-free rate. Defaults to 0.
            ewm (bool): Whether to use ewm (True) or rolling (False). Defaults to False
            alpha (float, optional): Alpha parameter for ewm. Defaults to None
            halflife (float, optional): Halflife parameter for ewm. Defaults to None
            bollinger_bands (bool): Whether to calculate bollinger bands or not. Defaults to False
            num_std (float): Number of standard deviations for bollinger bands. Defaults to 2.
            filename (str, optional): Path to save the figure. Defaults to None.
            fig (plotly.graph_objects.Figure | matplotlib.figure.Figure, optional): Existing figure to plot on.
                Used for overlaying plots or creating subplots. Defaults to None.
            subplot_idx (int | tuple[int, int], optional): Subplot position specifier.
                For Matplotlib: integer index.
                For Plotly: (row, col) tuple. Defaults to None.

        Returns:
            (plotly.graph_objects.Figure | matplotlib.figure.Figure): Moving average price of underlying asset.
                Returns Plotly figure if interactive=True, Matplotlib figure otherwise.
        """
        # get MA data
        data = self.rolling_stats(window=window, five_min=True if timeframe != '1d' else False,
                                r=r, ewm=ewm, alpha=alpha, halflife=halflife, 
                                bollinger_bands=bollinger_bands, num_std=num_std)
        
        if start_date is not None:
            data = data[data.index >= start_date]
        if end_date is not None:
            data = data[data.index <= end_date]
        if resample is not None:
            # aggregate parameters for resampling
            agg = {
                    'close_mean': 'last',
                    'adj_close_mean': 'last',
                    'close_std': 'mean',
                    'adj_close_std': 'mean',
                    'rets_mean': 'sum',
                    'log_rets_mean': 'sum',
                    'rets_std': 'mean',
                    'log_rets_std': 'mean',
                    'sharpe': 'mean'
                }

            boll_agg = {
                'bol_up': 'last',
                'bol_low': 'last'
            }

            if bollinger_bands:
                agg = agg | boll_agg

            data = data.resample(resample).agg(agg)

        data = data.dropna()

        # configure parameter text
        if alpha is not None:
            param = f'{alpha=}'
        elif halflife is not None:
            param = f'{halflife=}'
        else:
            param = f'{window=}'

        # static plot
        if not interactive:

            # create or use existing figure
            standalone = True
            if fig is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            else:
                ax = fig.axes[subplot_idx] if subplot_idx is not None else fig.gca()
                standalone = False

            plt.style.use('seaborn-v0_8')

            # Create the base plot with the main price line
            sns.lineplot(data=data, x=data.index, y='close_mean', 
                        color='#2962FF', linewidth=2, label=f'{self.ticker} MA ({param})',
                        ax=ax)

            if bollinger_bands:
                # Add the Bollinger Bands
                sns.lineplot(data=data, x=data.index, y='bol_up', 
                            color='#FF4081', linestyle='--', linewidth=1, label='Upper Band',
                            ax=ax)
                sns.lineplot(data=data, x=data.index, y='bol_low', 
                            color='#FF4081', linestyle='--', linewidth=1, label='Lower Band',
                            ax=ax)

                # Fill between the bands
                ax.fill_between(data.index, data['bol_low'], data['bol_up'], 
                                alpha=0.1, color='#2962FF')

            # Customize the plot
            ax.set_title(f'{self.ticker} Moving Average ({param}) {f'with Bollinger Bands ({num_std=})' if bollinger_bands else ''}', pad=20)
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Price ({self.currency})')

            def format_prices(x, p):
                if x >= 1e3:
                    return f'{x/1e3:.1f}K'
                else:
                    return f'{x:.1f}'

            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_prices))
            ax.legend(loc='upper left', framealpha=0.9)
            ax.grid(True, alpha=0.2)
            plt.xticks(rotation=0)

            if filename is not None:
                fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=(True if filename.endswith('.png') else False))

            if standalone:
                plt.show()

        # interactive plot
        else:
            # create or use existing figure
            standalone = False
            if fig is None:
                fig = go.Figure()
                standalone = True

            # Add main price line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['close_mean'],
                    line=dict(
                        color='#2962FF',
                        width=2,
                        dash='solid'
                    ),
                    name=f'{self.ticker} MA {param}'
                ),
                row=subplot_idx[0] if subplot_idx else None,
                col=subplot_idx[1] if subplot_idx else None
            )

            if bollinger_bands:
                # Add lower band
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['bol_low'],
                    name='Lower Band',
                    line=dict(color='#FF4081', width=1, dash='dash'),
                    mode='lines',
                    showlegend=True
                    ),
                    row=subplot_idx[0] if subplot_idx else None,
                    col=subplot_idx[1] if subplot_idx else None
                )

                # Add upper band with fill
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['bol_up'],
                    name='Upper Band',
                    line=dict(color='#FF4081', width=1, dash='dash'),
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(68, 68, 255, 0.1)',
                    showlegend=True
                    ),
                    row=subplot_idx[0] if subplot_idx else None,
                    col=subplot_idx[1] if subplot_idx else None
                )

            # update layout
            fig.update_layout(
                title=dict(
                    text=f'{self.ticker} Moving Average ({param}) {f'with Bollinger Bands ({num_std=})' if bollinger_bands else ''}',
                    x=0.5,  # Center the title
                    y=0.95
                ) if standalone else None,
                paper_bgcolor='white',
                plot_bgcolor='rgba(240,240,240,0.95)',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    title=None,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    title=f'Price ({self.currency})',
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                hovermode='x unified'
            )

            if not standalone:
                fig.update_yaxes(
                    title_text=f'Price ({self.currency})',
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )
                fig.update_xaxes(
                    title_text=f'{self.ticker} Moving Average {f'with Bollinger Bands ({num_std=})' if bollinger_bands else ''}', 
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )

            if standalone:
                fig.show()

            if filename is not None:
                fig.write_image(filename)

        return fig

    def plot_returns_dist(self, *, log_rets: bool = False, bins: int = 100, interactive: bool = True, 
                        filename: Optional[str] = None, fig: Optional[go.Figure | Figure] = None, 
                        subplot_idx: Optional[int | tuple[int, int]] = None, show_stats: bool = True) -> go.Figure | Figure:
        """Plots the returns distribution histogram of the underlying asset.

        Args:
            log_rets (bool): Whether to use log returns (True) or normal returns (False)
                Defaults to False
            interactive (bool, optional): Whether to create an interactive Plotly graph (True) 
                or static Matplotlib graph (False). Defaults to True.
            filename (str, optional): Path to save the figure. Defaults to None.
            fig (plotly.graph_objects.Figure | matplotlib.figure.Figure, optional): Existing figure to plot on.
                Used for overlaying plots or creating subplots. Defaults to None.
            subplot_idx (int | tuple[int, int], optional): Subplot position specifier.
                For Matplotlib: integer index.
                For Plotly: (row, col) tuple. Defaults to None.
            show_stats (bool): Whether or not to show distribution stats. Defaults to True

        Returns:
            (plotly.graph_objects.Figure | matplotlib.figure.Figure): Returns distribution histogram of underlying asset.
                Returns Plotly figure if interactive=True, Matplotlib figure otherwise.
        """

        data = self.daily['log_rets'] if log_rets else self.daily['rets']
        data = data.dropna()

        # Calculate statistics
        stats_text = (
            f'Mean: {np.mean(data):.4f}{'\n' if not interactive else '<br>'}'
            f'Std Dev: {np.std(data):.4f}{'\n' if not interactive else '<br>'}'
            f'Skewness: {stats.skew(data):.4f}{'\n' if not interactive else '<br>'}'
            f'Kurtosis: {stats.kurtosis(data):.4f}'
        )

        # static plot
        if not interactive:

            plt.style.use('seaborn-v0_8')

            # create or use existing figure
            if fig is None:
                standalone = True
                fig, ax = plt.subplots(figsize=(12, 7))
            else:
                standalone = False
                ax = fig.axes[subplot_idx] if subplot_idx is not None else fig.gca()

            # Create histogram
            sns.histplot(data, 
                        bins=bins,
                        color='#2E86C1',
                        alpha=0.7,
                        edgecolor='white',
                        ax=ax)

            # Customize the plot
            ax.set_title(f'{self.ticker} Returns Distribution', fontsize=14, pad=15, fontweight='bold')
            ax.set_xlabel('Returns', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if show_stats:
                # Add stats box
                plt.text(0.75, 0.85, stats_text,
                        transform=ax.transAxes,
                        bbox=dict(
                            facecolor='white',
                            edgecolor='#2E86C1',
                            alpha=0.9,
                            boxstyle='round,pad=0.5'
                        ),
                        fontsize=10,
                        verticalalignment='top')

            ax.grid(True, alpha=0.3, linestyle='--')
            fig.subplots_adjust(hspace=0.4)

            if filename is not None:
                    fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=(True if filename.endswith('.png') else False))

            if standalone:
                plt.show()

        # interactive plot
        else:
            # create or use existing figure
            standalone = False
            if fig is None:
                fig = go.Figure()
                standalone = True

            bins = np.linspace(data.min(), data.max(), bins + 1)

            # create histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    xbins=dict(
                        start=bins[0],
                        end=bins[-1],
                        size=(bins[1] - bins[0])  # Forces exact bin width
                    ),
                    name=f'{self.ticker} Returns Distribution'
                ),
                row=subplot_idx[0] if subplot_idx else None,
                col=subplot_idx[1] if subplot_idx else None
            )

            # configure statsbox position
            xref = 'paper'
            yref = 'paper'

            if subplot_idx:
                xref = f'x{subplot_idx[0] if subplot_idx[0] != 1 else ''} domain'
                yref = f'y{subplot_idx[0] if subplot_idx[0] != 1 else ''} domain'

            if show_stats:
                fig.add_annotation(
                    x=0.95,
                    y=0.95,
                    xref=xref,
                    yref=yref,
                    text=stats_text,
                    showarrow=False,
                    font=dict(size=10),
                    align='left',
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    xanchor='right',
                    yanchor='top'
                )

            fig.update_layout(
                yaxis=dict(
                    range=[0, None],
                    rangemode='nonnegative'
                ),
                bargap=0.05
            )

            if standalone:
                fig.update_layout(
                        title=f'{self.ticker} {'Log' if log_rets else ''} Returns Distribution',
                        xaxis_title='Returns',
                        yaxis_title=f'Count'
                    )

                fig.show()
            else:
                fig.update_yaxes(
                    title_text='Count',
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )
                fig.update_xaxes(
                    title_text=f'{self.ticker} {'Log' if log_rets else ''} Returns Distribution', 
                    row=subplot_idx[0] if subplot_idx else None, 
                    col=subplot_idx[1] if subplot_idx else None
                )

            if filename is not None:
                fig.write_image(filename)

        return fig

    def resample(self, period: str, five_min: bool = False) -> DataFrame:
        """Resamples the asset data

        Args:
            period (str): Resampling frequency in pandas format (e.g., 'B' for business day).
            five_min (bool): Whether to use 5min data (True) or daily data (False). Defaults to False

        Returns:
            pandas.core.frame.DataFrame: df of resampled data
        """
        if five_min:
            data = self.five_minute
        else:
            data = self.daily

        data = data.resample(period).agg({
            'open': 'first',     # First price of the month
            'high': 'max',       # Highest price of the month
            'low': 'min',        # Lowest price of the month
            'close': 'last',     # Last price of the month
            'adj_close': 'last', # Last adjusted price of the month
            'volume': 'sum',     # Total volume for the month
            'log_rets': 'sum',   # Sum of log returns
            'rets': 'sum'        # Sum of returns
        })

        data = data.dropna()

        return data

    def rolling_stats(self, *, window: int = 20, five_min: bool = False, r: float = 0., ewm: bool = False, 
                       alpha: Optional[float] = None, halflife: Optional[float] = None, bollinger_bands: bool = False, 
                       num_std: float = 2., sharpe_ratio: bool = False) -> DataFrame:
        """Calculate rolling stats for asset data

        Args:
            window (int): Rolling window. Defaults to 20
            five_min (bool): Whether to use 5min data (True) or daily data (False). Defaults to False
            r (float): risk-free rate. Defaults to 0.
            ewm (bool): Whether to use ewm (True) or rolling (False). Defaults to False
            alpha (float, optional): Alpha parameter for ewm. Defaults to None
            halflife (float, optional): Halflife parameter for ewm. Defaults to None
            bollinger_bands (bool): Whether to calculate bollinger bands or not. Defaults to False
            num_std (float): Number of standard deviations for bollinger bands. Defaults to 2.
            sharpe_ratio (bool): Whether or not to calculate rolling Sharpe ratio. Defaults to False

        Returns:
            pandas.core.frame.DataFrame: df of the rolling data with mean and standard deviation
                for close prices and returns as well as sharpe ratio if specified
        """

        if five_min:
            data = self.five_minute
            if self.asset_type == 'cryptocurrency':
                annualization_factor = 252 * 24 * 12  # 24/7 trading
            else:
                annualization_factor = 252 * 78  # Assuming ~78 5-min periods per day
        else:
            data = self.daily
            annualization_factor = 252  # Trading days in a year

        roll_df = pd.DataFrame()

        param = {}
        if ewm:
            if alpha is not None:
                param['alpha'] = alpha
            elif halflife is not None:
                param['halflife'] = halflife
            else:
                param['span'] = window

        # calculate mean and std rolling data for close prices and returns
        cols = ['close', 'adj_close', 'rets', 'log_rets']
        for col in cols:
            if ewm:
                roll_df[f'{col}_mean'] = data[f'{col}'].ewm(**param).mean()
                roll_df[f'{col}_std'] = data[f'{col}'].ewm(**param).std()
            else:
                roll_df[f'{col}_mean'] = data[f'{col}'].rolling(window=window).mean()
                roll_df[f'{col}_std'] = data[f'{col}'].rolling(window=window).std()

        roll_df = roll_df.dropna()

        # Calculate annualized Sharpe ratio
        if sharpe_ratio:
            daily_rf_rate = (1 + r) ** (1/annualization_factor) - 1
            excess_returns = roll_df['rets_mean'] - daily_rf_rate
            roll_df['sharpe'] = (excess_returns / roll_df['rets_std']) * np.sqrt(annualization_factor)

        # add bollinger bands
        if bollinger_bands:
            roll_df = self._add_bollinger_bands(roll_df, num_std=num_std)

        return roll_df

    @property
    def stats(self) -> dict:
        """Basic statistics for the underlying asset
           - Return statistics
           - Price statistics
           - Distribution statistics

        Returns:
        dict: dictionary containing statistics of the asset
        """
        stats = {}

        # return statistics
        stats['returns'] = {
            'total_return': (self.daily['close'].iloc[-1] / 
                            self.daily['close'].iloc[0]) - 1,
            'daily_mean': self.daily['rets'].mean(),
            'daily_std': self.daily['rets'].std(),
            'daily_median': self.daily['rets'].median(),
            'annualized_vol': self.daily['rets'].std() * np.sqrt(252)
        }

        stats['returns'] = {k: float(v) for k, v in stats['returns'].items()}

        # price statistics
        stats['price'] = {
            'high': self.daily['high'].max(),
            'low': self.daily['low'].min(),
            '52w_high': self.daily[self.daily.index >= datetime.now() - pd.Timedelta(weeks=52)]['high'].max(),
            '52w_low': self.daily[self.daily.index >= datetime.now() - pd.Timedelta(weeks=52)]['low'].max(),
            'current': self.daily['close'].iloc[-1]
        }

        stats['price'] = {k: float(v) for k, v in stats['price'].items()}

        # distribution statistics
        stats['distribution'] = {
            'skewness': self.daily['rets'].skew(),
            'kurtosis': self.daily['rets'].kurtosis()
        }

        stats['distribution'] = {k: float(v) for k, v in stats['distribution'].items()}

        # TODO:
        # add risk statistics: var95, cvar95, max_drawdown, drawdown_period, downside volatility
        # add risk_adjusted returns:  sharpe ratio, sortino ratio, calmar ratio
        # add distribution: normality test, jarque_bera
        # add trading stats: pos/neg days, best/worst day, avg up/down day, vol_mean, vol_std

        return stats

    def _add_bollinger_bands(self, df: DataFrame, num_std: float = 2.) -> DataFrame:
        """Helper method to calculate bollinger bands
        
        Args:
            df (pandas.core.frame.DataFrame): df to add bollinger bands onto
            num_std (float): Number of standard deviations away for bollinger bands

        Returns:
            pandas.core.frame.DataFrame: original df with bollinger bands columns
            """
        df['bol_up'] = df['close_mean'] + num_std * df['close_std']
        df['bol_low'] = df['close_mean'] - num_std * df['close_std']

        return df

# TODO:
# backtest SMA strategy
# optimize SMA window
# plot more diagrams
# add more technical indicators (RSI, MACD, ATR)
# simple default dashboard
