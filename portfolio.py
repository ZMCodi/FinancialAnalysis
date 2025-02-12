import numpy as np
import pandas as pd
from assets import Asset
from collections import Counter, defaultdict, namedtuple
import psycopg as pg
from config import DB_CONFIG
import datetime

DateLike = str | datetime.datetime | datetime.date | pd.Timestamp

# store how many shares e.g. NVDA 30 shares
# buy/sell: give date and either shares or price
# Portfolio({AAPL: {'shares': 20, 'avg_price': 100}, NVDA: {'shares': 15, 'avg_price': 20}})
# store total money invested and num of shares for each asset


class Portfolio:

    transaction = namedtuple('transaction', ['type', 'asset', 'shares', 'value', 'date'])

    def __init__(self, assets: dict | None = None, currency: str | None = None):
        self.holdings = defaultdict(float)
        self.deposits = 0
        self.currency = 'USD' if currency is None else currency
        self.cost_bases = defaultdict(float)
        self.transactions = []
        self.assets = []
        self.forex_cache = {}

        if assets:  # Only process if assets provided
            self.assets.extend([Asset(ast.ticker) for ast in assets])  # store copy of assets
            self.holdings.update({k: v['shares'] for k, v in assets.items()})
            self.cost_bases.update({k: v['avg_price'] for k, v in assets.items()})
            self.deposits = sum(ast['shares'] * ast['avg_price'] for ast in assets.values())

            if currency is None:
                self.currency = Counter((ast.currency for ast in assets)).most_common()[0][0]
            else:
                self.currency = currency

            for ast in self.assets:
                if ast.currency != self.currency:
                    self._convert(ast)

    def _convert(self, asset: Asset):
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

    def buy(self, asset: Asset, *, shares: float | None = None, value: float | None = None, 
            date: DateLike | None = None):
        if date is None:
            date = datetime.date.today()
        date = date.strftime('%Y-%m-%d')

        if asset not in self.assets:
            ast = Asset(asset.ticker)  # create copy
            if ast.currency != self.currency:
                self._convert(ast)
            self.assets.append(ast)

        # get price at buy
        idx = self.assets.index(asset)
        ast = self.assets[idx]
        price = float(ast.daily.loc[date, 'close'])

        if shares is None:
            # get shares from value / price at date
            shares = value / price

        if value is None:
            # get value from shares * price at date
            value = shares * price

        # update portfolio values
        self.transactions.append(self.transaction('BUY', asset, shares, value, date))
        old_cost_basis = self.cost_bases[asset]
        self.holdings[asset] += shares
        self.cost_bases[asset] = (old_cost_basis + value) / self.holdings[asset]
        self.deposits += value

    def sell(self, asset: Asset, *, shares: float | None = None, value: float | None = None, 
            date: DateLike | None = None):
        if date is None:
            date = datetime.date.today()
        date = date.strftime('%Y-%m-%d')

        # get price at sell
        idx = self.assets.index(asset)
        ast = self.assets[idx]
        price = float(ast.daily.loc[date, 'close'])

        if shares is None:
            # get shares from value / price at date
            shares = value / price

        if value is None:
            # get value from shares * price at date
            value = shares * price

        self.transactions.append(self.transaction('SELL', asset, shares, value, date))
        self.holdings[asset] -= shares
        self.deposits -= shares * self.cost_bases[asset]

    def rebalance(self):
        pass

    @property
    def stats(self):
        pass

    @property
    def PnL(self):
        pass

    @property
    def returns(self):
        pass

    @property
    def volatility(self):
        pass

    @property
    def sharpe_ratio(self):
        pass

    @property
    def VaR(self):
        pass

    def correlation_matrix(self):
        pass

    def save(self, name):
        pass

    @classmethod
    def load(cls, name):
        pass

    @classmethod
    def report(cls, name):
        pass
