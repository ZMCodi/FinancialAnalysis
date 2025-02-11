import numpy as np
import pandas as pd
from assets import Asset
from collections import Counter, defaultdict

# store how many shares e.g. NVDA 30 shares
# buy/sell: give date and either shares or price
# Portfolio({AAPL: {'shares': 20, 'avg_price': 100}, NVDA: {'shares': 15, 'avg_price': 20}})
# store total money invested and num of shares for each asset


class Portfolio:

    def __init__(self, assets: dict | None = None, currency: str | None = None):
        self.holdings = defaultdict(int)
        self.deposits = 0
        self.currency = 'USD' if currency is None else currency
        self.cost_basis = defaultdict(float)

        if assets:  # Only process if assets provided
            self.holdings.update({k: v['shares'] for k, v in assets.items()})
            self.cost_basis.update({k: v['avg_price'] for k, v in assets.items()})
            self.deposits = sum(ast['shares'] * ast['avg_price'] for ast in assets.values())

            if currency is None:
                self.currency = Counter((ast.currency for ast in assets)).most_common()[0][0]
            else:
                self.currency = currency

            for ast in self.holdings.keys():
                if ast.currency != self.currency:
                    self._convert(ast)


    def _convert(self, asset):
        pass

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
