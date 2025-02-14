import numpy as np
import pandas as pd
from assets import Asset
from collections import Counter, defaultdict, namedtuple
import psycopg as pg
from config import DB_CONFIG
import datetime
import plotly.graph_objects as go
from scipy import stats

DateLike = str | datetime.datetime | datetime.date | pd.Timestamp

# store how many shares e.g. NVDA 30 shares
# buy/sell: give date and either shares or price
# Portfolio({AAPL: {'shares': 20, 'avg_price': 100}, NVDA: {'shares': 15, 'avg_price': 20}})
# store total money invested and num of shares for each asset


class Portfolio:

    transaction = namedtuple('transaction', ['type', 'asset', 'shares', 'value', 'date'])

    def __init__(self, assets: dict | None = None, currency: str | None = None, r: float = 0.02):
        self.holdings = defaultdict(float)
        self.deposits = 0
        self.currency = 'USD' if currency is None else currency
        self.cost_bases = defaultdict(float)
        self.transactions = []
        self.assets = []
        self.forex_cache = {}
        self.r = r

        if assets:  # Only process if assets provided
            self.assets.extend([Asset(ast.ticker) for ast in assets.keys()])  # store copy of assets

            if currency is None:
                self.currency = Counter((ast.currency for ast in assets)).most_common()[0][0]
            else:
                self.currency = currency

            for ast in self.assets:
                if ast.currency != self.currency:
                    self._convert_ast(ast)

            for ast in self.assets:
                self.holdings[ast] = assets[ast]['shares']
                if ast.currency != self.currency:
                    avg_price = self._convert_price(assets[ast]['avg_price'], ast.currency)
                else:
                    avg_price = assets[ast]['avg_price']

                self.cost_bases[ast] = avg_price

            self.deposits = sum(ast['shares'] * ast['avg_price'] for ast in assets.values())

    def _convert_price(self, price: float, currency: str, date: DateLike | None = None) -> float:
        date = self._parse_date(date)[:10]

        f = currency
        t = self.currency
        key = f'{f}/{t}'
        while True:
            if date in self.forex_cache[key].index:
                rate = self.forex_cache[key].loc[date, 'close']
                break
            else:
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
                date_obj -= datetime.timedelta(days=1)
                date = date_obj.strftime('%Y-%m-%d')

        return float(price * rate)

    def _convert_ast(self, asset: Asset) -> None:
        f = asset.currency
        t = self.currency
        key = f'{f}/{t}'
        if key not in self.forex_cache:
            with pg.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT currency_pair, date, close FROM daily_forex WHERE currency_pair = %s", (key,))
                    forex = cur.fetchall()

            forex = pd.DataFrame(forex, columns=['pair', 'date', 'close']).set_index('date')
            forex.index = pd.to_datetime(forex.index)
            forex = forex.sort_index()
            forex['close'] = forex['close'].astype(float)
            self.forex_cache[key] = forex

        forex = self.forex_cache[key]

        for df in [asset.daily, asset.five_minute]:
            frx = forex.reindex_like(df, method='ffill')[['close']]
            df[['open', 'high', 'low', 'close', 'adj_close']] = df[['open', 'high', 'low', 'close', 'adj_close']].mul(frx['close'], axis=0)
            df['log_rets'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
            df['rets'] = df['adj_close'].pct_change()

    def _parse_date(self, date: DateLike | None = None) -> str:
        if date is None:
            date = datetime.date.today()

        if isinstance(date, pd.Timestamp):
            # Convert pandas.Timestamp to datetime or date
            date = date.to_pydatetime() if not pd.isna(date) else date.date()

        if isinstance(date, datetime.datetime):
            date = date.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(date, datetime.date):  # datetime.date object
            date = date.strftime('%Y-%m-%d')

        return date
    
    def _get_price(self,  ast: Asset, date: str) -> float:
        date = date[:10]
        while True:
            if date in ast.daily.index:
                price = ast.daily.loc[date, 'adj_close']
                return price
            else:
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
                date_obj -= datetime.timedelta(days=1)
                date = date_obj.strftime('%Y-%m-%d')

    def buy(self, asset: Asset, *, shares: float | None = None, value: float | None = None, 
            date: DateLike | None = None, currency: str | None = None) -> None:

        date = self._parse_date(date)

        if currency is None:
            currency = asset.currency

        if asset not in self.assets:
            ast = Asset(asset.ticker)  # create copy
            if ast.currency != self.currency:
                self._convert_ast(ast)
            self.assets.append(ast)

        # get price at buy
        idx = self.assets.index(asset)
        ast = self.assets[idx]

        if value is not None:
            if currency != self.currency:
                value = self._convert_price(value, currency, date)

        if shares is None:
            # get shares from value / price at date
            price = self._get_price(ast, date)
            shares = value / price

        if value is None:
            # get value from shares * price at date
            price = self._get_price(ast, date)
            value = shares * price

        # update portfolio values
        self.transactions.append(self.transaction('BUY', ast, float(shares), float(value), date))
        old_cost_basis = self.cost_bases[ast] * self.holdings[ast]
        self.holdings[ast] += float(shares)
        self.cost_bases[ast] = (old_cost_basis + value) / self.holdings[ast]
        self.deposits += float(value)

    def sell(self, asset: Asset, *, shares: float | None = None, value: float | None = None, 
            date: DateLike | None = None, currency: str | None = None) -> None:

        date = self._parse_date(date)

        if currency is None:
            currency = self.currency

        # get price at sell
        idx = self.assets.index(asset)
        ast = self.assets[idx]

        if value is not None:
            if currency != self.currency:
                value = self._convert_price(value, currency, date)

        if shares is None:
            # get shares from value / price at date
            price = self._get_price(ast, date)
            shares = value / price

        if value is None:
            # get value from shares * price at date
            price = self._get_price(ast, date)
            value = shares * price

        self.transactions.append(self.transaction('SELL', ast, float(shares), float(value), date))
        self.holdings[ast] -= shares
        self.deposits -= value
        if self.holdings[ast] < 1e-8:
            del self.holdings[ast]
            del self.cost_bases[ast]
            del self.assets[idx]

    def pie_chart(self):
        fig = go.Figure()
        data = {k.ticker: v for k, v in self.weights.items()}

        # Create custom text array - empty string for small values
        text = [f'{p:.1f}%' if p >= 5 else '' for p in data.values()]

        fig.add_trace(go.Pie(
            labels=list(data.keys()),
            values=list(data.values()),
            text=text,
            textinfo='text',
            hoverinfo='label+percent',
            textposition='auto',
        ))

        fig.update_layout(
            title='Porfolio Holdings',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="right",
                x=1
            )
        )

        fig.show()
        return fig
    
    def returns_dist(self, bins: int = 100, show_stats: bool = True):
        data = self.returns.dropna()
        fig = go.Figure()

        # Calculate statistics
        stats_text = (
            f'Mean: {np.mean(data):.4f}<br>'
            f'Std Dev: {np.std(data):.4f}<br>'
            f'Skewness: {stats.skew(data):.4f}<br>'
            f'Kurtosis: {stats.kurtosis(data):.4f}'
        )

        bins = np.linspace(data.min(), data.max(), bins + 1)

        fig.add_trace(
            go.Histogram(
                x=data,
                xbins=dict(
                    start=bins[0],
                    end=bins[-1],
                    size=(bins[1] - bins[0])
                ),
                name='Portfolio Returns Distribution'
            )
        )

        xref = 'paper'
        yref = 'paper'

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
                bargap=0.05,
                title='Portfolio Returns Distribution',
                xaxis_title='Returns',
                yaxis_title='Count'
            )
        
        fig.show()

        return fig

    def rebalance(self):
        pass

    @property
    def stats(self):
        pass

    @property
    def total_returns(self) -> float:
        return self.get_pnl() / self.deposits

    @property
    def returns(self):

        if not self.transactions:
            return pd.Series()

        # Sort transactions by date
        sorted_transactions = sorted(self.transactions, key=lambda x: x.date)

        # Create a DataFrame of holdings changes
        holdings_changes = {}
        current_holdings = defaultdict(float)

        for t in sorted_transactions:
            if t.type == 'BUY':
                current_holdings[t.asset] += t.shares
            else:  # SELL
                current_holdings[t.asset] -= t.shares
            holdings_changes[t.date[:10]] = dict(current_holdings)

        # Convert to DataFrame and forward fill
        holdings_df = pd.DataFrame.from_dict(holdings_changes, orient='index').fillna(0)
        holdings_df.index = pd.to_datetime(holdings_df.index)
        holdings_df = holdings_df.reindex(
            pd.date_range(start=holdings_df.index[0].date(), end=pd.Timestamp.today())
        ).ffill()

        prices = pd.DataFrame(index=holdings_df.index)
        for ast in holdings_df.columns:
            prices[ast] = ast.daily['adj_close']
        prices = prices.ffill()

        portfolio_values = holdings_df.mul(prices).sum(axis=1)
        portfolio_values = portfolio_values[portfolio_values != 0]
        rets = portfolio_values.pct_change()
        return rets[rets != 0]

    def get_pnl(self, date: DateLike | None = None) -> float:
        date = self._parse_date(date)[:10]

        curr_value = self.get_value(date)
        return curr_value - self.deposits

    def get_value(self, date: DateLike | None = None) -> float:
        date = self._parse_date(date)

        return sum(self.holdings_value(date).values())

    def holdings_pnl(self, date: DateLike | None = None) -> dict:
        date = self._parse_date(date)
        date = date.strftime('%Y-%m-%d') if type(date) != str else date[:10]
        value = self.holdings_value(date)
        return {ast: float(value[ast] - (self.holdings[ast] * self.cost_bases[ast]))
                for ast in self.assets}

    def holdings_value(self, date: DateLike | None = None) -> dict:
        date = self._parse_date(date)[:10]

        return {asset: float(self._get_price(asset, date) * shares)
                for asset, shares in self.holdings.items()}

    @property
    def volatility(self):
        if not self.weights:
            return 0

        stock_weight = sum(v for k, v in self.weights.items() if k.asset_type != 'cryptocurrency')
        crypto_weight = sum(v for k, v in self.weights.items() if k.asset_type == 'cryptocurrency')

        # Weight the annualization factor
        ann_factor = (stock_weight * 252) + (crypto_weight * 365)

        daily_vol = self.returns.std()
        return daily_vol * np.sqrt(ann_factor)

    @property
    def weights(self):
        return {k: v / sum(self.holdings_value().values()) for k, v in self.holdings_value().items()}

    @property
    def sharpe_ratio(self):

        if not self.weights:
            return 0

        # Calculate weighted annualization factor
        stock_weight = sum(v for k, v in self.weights.items() if k.asset_type != 'cryptocurrency')
        crypto_weight = sum(v for k, v in self.weights.items() if k.asset_type == 'cryptocurrency')
        ann_factor = (stock_weight * 252) + (crypto_weight * 365)

        # Convert annual risk-free rate to daily using weighted factor
        daily_rf = self.r / ann_factor

        # Calculate excess returns
        excess_returns = self.returns - daily_rf

        # Annualize mean excess returns
        mean_excess_returns = excess_returns.mean() * ann_factor

        vol = self.volatility
        if vol == 0:
            return 0

        return mean_excess_returns / vol

    def VaR(self, confidence: float = 0.95):
        return float(np.abs(self.returns.quantile(1 - confidence) * self.get_value()))

    def correlation_matrix(self):
        df = pd.DataFrame()
        for ast in self.assets:
            df[ast.ticker] = ast.daily['rets']

        corr_matrix = df.corr()
        corr_matrix_masked = np.tril(corr_matrix, k=-1)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix_masked,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            zmin=-1,
            zmax=1,
            colorscale='RdBu',
            hoverongaps=False
        ))

        # Update layout
        fig.update_layout(
            title='Asset Correlation Matrix',
            height=800,
            xaxis=dict(
                tickangle=-90,
                side='bottom'
            ),
            yaxis=dict(
                autorange='reversed'
            )
        )

        fig.show()
        return fig

    def save(self, name):
        pass

    @classmethod
    def load(cls, name):
        pass

    @classmethod
    def report(cls, name):
        pass
