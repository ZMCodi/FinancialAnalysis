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
        if date is None:
            date = datetime.date.today() - datetime.timedelta(days=1)
        date = date.strftime('%Y-%m-%d') if type(date) != str else date[:10]

        f = currency
        t = self.currency
        key = f'{f}/{t}'
        rate = self.forex_cache[key].loc[date, 'close']
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
            date = datetime.datetime.now() - datetime.timedelta(days=1)

        if isinstance(date, pd.Timestamp):
            # Convert pandas.Timestamp to datetime or date
            date = date.to_pydatetime() if not pd.isna(date) else date.date()

        if isinstance(date, datetime.datetime):
            date = date.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(date, datetime.date):  # datetime.date object
            date = date.strftime('%Y-%m-%d')

        return date

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
            price = float(ast.daily.loc[date, 'adj_close'])
            shares = value / price

        if value is None:
            # get value from shares * price at date
            price = float(ast.daily.loc[date, 'adj_close'])
            value = shares * price

        # update portfolio values
        self.transactions.append(self.transaction('BUY', ast, shares, value, date))
        old_cost_basis = self.cost_bases[ast] * self.holdings[ast]
        self.holdings[ast] += shares
        self.cost_bases[ast] = (old_cost_basis + value) / self.holdings[ast]
        self.deposits += value

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
            price = float(ast.daily.loc[date, 'adj_close'])
            shares = value / price

        if value is None:
            # get value from shares * price at date
            price = float(ast.daily.loc[date, 'adj_close'])
            value = shares * price

        self.transactions.append(self.transaction('SELL', ast, shares, value, date))
        self.holdings[ast] -= shares
        self.deposits -= value
        if self.holdings[ast] < 1e-8:
            del self.holdings[ast]
            del self.cost_bases[ast]
            del self.assets[idx]

    def rebalance(self):
        pass

    @property
    def stats(self):
        pass

    def get_pnl(self, date: DateLike | None = None) -> float:
        date = self._parse_date(date)
        date = date.strftime('%Y-%m-%d') if type(date) != str else date[:10]

        curr_value = self.get_value(date)
        return curr_value - self.deposits

    @property
    def returns(self):
        pass

    def get_value(self, date: DateLike | None = None) -> float:
        date = self._parse_date(date)
        date = date.strftime('%Y-%m-%d') if type(date) != str else date[:10]

        return sum(float(asset.daily.loc[date, 'adj_close']) * shares 
                         for asset, shares in self.holdings.items())

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
