import numpy as np
import pandas as pd
from assets import Asset
from collections import Counter, defaultdict
import psycopg as pg
from config import DB_CONFIG

# store how many shares e.g. NVDA 30 shares
# buy/sell: give date and either shares or price
# Portfolio({AAPL: {'shares': 20, 'avg_price': 100}, NVDA: {'shares': 15, 'avg_price': 20}})
# store total money invested and num of shares for each asset


class Portfolio:

    def __init__(self, assets: dict | None = None, currency: str | None = None):
        self.holdings = defaultdict(float)
        self.deposits = 0
        self.currency = 'USD' if currency is None else currency
        self.cost_basis = defaultdict(float)
        self.transactions = []
        self.assets = []

        if assets:  # Only process if assets provided
            self.assets.extend([Asset(ast.ticker) for ast in assets])  # store copy of assets
            self.holdings.update({k: v['shares'] for k, v in assets.items()})
            self.cost_basis.update({k: v['avg_price'] for k, v in assets.items()})
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
        with pg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT currency_pair, date, close FROM daily_forex WHERE currency_pair = %s", (f'{f}/{t}',))
                forex = cur.fetchall()

        forex = pd.DataFrame(forex, columns=['pair', 'date', 'close']).set_index('date')
        forex.index = pd.to_datetime(forex.index)
        forex = forex.sort_index()
        forex['close'] = forex['close'].astype(float)

        for df in [asset.daily, asset.five_minute]:
            frx = forex.reindex_like(df, method='ffill')[['close']]
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].mul(frx['close'], axis=0)

    def buy(self, asset, value):
        pass

    def sell(self, asset, value):
        pass

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
