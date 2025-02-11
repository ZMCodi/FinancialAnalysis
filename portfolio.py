import numpy as np
import pandas as pd
from assets import Asset
from typing import Iterable

class Portfolio:

    def __init__(self, assets: Iterable[Asset], weights):
        pass

    def _convert(self):
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
