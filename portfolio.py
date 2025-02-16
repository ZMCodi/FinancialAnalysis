import numpy as np
import pandas as pd
from assets import Asset
from collections import Counter, defaultdict, namedtuple
import psycopg as pg
from config import DB_CONFIG
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
        self.cash = 0.0

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
                self.cash = assets.get('Cash', 0)

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
        if f == t:
            return
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
            df['rets'] = df['adj_close'].pct_change(fill_method=None)

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

    def deposit(self, value: float, currency: str | None = None, date: DateLike | None = None):
        date = self._parse_date(date)

        if currency != self.currency:
            value = self._convert_price(value, currency, date)

        self.transactions.append(self.transaction('DEPOSIT', 'Cash', 0.0, float(value), date))
        self.cash += value

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
        if self.cash - value < 0:
            raise ValueError('Not enough money')

        self.transactions.append(self.transaction('BUY', ast, float(shares), float(value), date))
        old_cost_basis = self.cost_bases[ast] * self.holdings[ast]
        self.holdings[ast] += float(shares)
        self.cost_bases[ast] = (old_cost_basis + value) / self.holdings[ast]
        self.cash -= value

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

        self.holdings[ast] -= float(shares)
        self.cash += float(value)
        if self.holdings[ast] < 1e-8:
            del self.holdings[ast]
            del self.assets[idx]

    def pie_chart(self, data: dict, title: str):
        fig = go.Figure()

        # Create custom text array - empty string for small values
        text = [f'{p*100:.1f}%' if p >= 0.05 else '' for p in data.values()]

        fig.add_trace(go.Pie(
            labels=list(data.keys()),
            values=list(data.values()),
            text=text,
            textinfo='text',
            hoverinfo='label+percent',
            textposition='auto',
        ))

        fig.update_layout(
            title=title,
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

    def holdings_chart(self):
        weights = self.weights
        data = {k.ticker: v for k, v in weights.items()}
        return self.pie_chart(data, 'Portfolio Holdings')

    def asset_type_exposure(self):
        data = defaultdict(float)
        weights = self.weights
        for ast, weight in weights.items():
            data[ast.asset_type] += weight

        return self.pie_chart(data, 'Asset Type Exposure')

    def sector_exposure(self):
        data = defaultdict(float)
        weights = self.weights
        for ast, weight in weights.items():
            if ast.sector is not None:
                data[ast.sector] += weight

        data = {k: v / sum(data.values()) for k, v in data.items()}

        return self.pie_chart(data, 'Sector Exposure')

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
        return np.exp(self.log_returns.sum())

    @property
    def trading_returns(self) -> float:
        """Trading return as a decimal: trading P&L divided by cost basis."""
        total_cost_basis = sum(self.holdings[ast] * self.cost_bases[ast] for ast in self.assets)
        return self.trading_pnl() / total_cost_basis if total_cost_basis else 0.0


    def _returns_helper(self):

        if not self.transactions:
            return pd.Series()

        # Sort transactions by date
        sorted_transactions = sorted(self.transactions, key=lambda x: x.date)

        # Create a DataFrame of holdings changes
        holdings_changes = {}
        current_holdings = defaultdict(float)
        running_deposit = defaultdict(float)
        cash = defaultdict(float)

        has_crypto = False
        for t in sorted_transactions:
            if type(t.asset) != str:
                if t.asset.asset_type == 'Cryptocurrency':
                    has_crypto = True
            if t.type == 'BUY':
                current_holdings[t.asset] += t.shares
                cash[t.date] -= t.value
            elif t.type == 'SELL':
                current_holdings[t.asset] -= t.shares
                cash[t.date] += t.value
            elif t.type == 'DEPOSIT':
                running_deposit[t.date] += t.value
                cash[t.date] += t.value
            holdings_changes[t.date[:10]] = dict(current_holdings)

        running_deposit = pd.Series(running_deposit)
        cash = pd.Series(cash)

        # Convert to DataFrame and forward fill
        holdings_df = pd.DataFrame.from_dict(holdings_changes, orient='index').fillna(0)
        holdings_df.index = pd.to_datetime(holdings_df.index)
        holdings_df = holdings_df.reindex(
            pd.date_range(start=holdings_df.index[0].date(), end=pd.Timestamp.today(), freq=('D' if has_crypto else 'B'))
        ).ffill()

        running_deposit.index = pd.to_datetime(running_deposit.index)
        cash.index = pd.to_datetime(cash.index)
        running_deposit = running_deposit.reindex(holdings_df.index).fillna(0).cumsum()
        cash = cash.reindex(holdings_df.index).fillna(0).cumsum()


        prices = pd.DataFrame(index=holdings_df.index)
        for ast in holdings_df.columns:
            prices[ast] = ast.daily['adj_close']
        prices = prices.ffill()

        portfolio_values = holdings_df.mul(prices).sum(axis=1)

        return running_deposit, cash, portfolio_values

    @property
    def pnls(self):
        deps, cash, values = self._returns_helper()
        return (values + cash - deps).diff()

    @property
    def returns(self):
        deps, cash, values = self._returns_helper()
        rets = (values + cash / deps)
        return rets.pct_change()

    @property
    def log_returns(self):
        rets = self.returns
        return np.log1p(rets)

    @property
    def realized_pnl(self) -> float:
        """Profit from sold shares: (sale proceeds) - (cost basis of sold shares)."""
        realized = 0.0
        for t in self.transactions:
            if t.type == 'SELL':
                # Find total cost basis of the sold shares
                cost_basis = t.shares * self.cost_bases[t.asset]
                realized += (t.value - cost_basis)
        return realized

    @property
    def unrealized_pnl(self) -> float:
        """Profit from current holdings: (current value) - (cost basis of remaining shares)."""
        return sum(self.holdings_pnl().values())

    def trading_pnl(self) -> float:
        """Total trading P&L: realized + unrealized."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_deposits(self):
        return sum(t.value for t in self.transactions if t.type == 'DEPOSIT')

    def investment_pnl(self, date: DateLike | None = None) -> float:
        """Total portfolio PnL at date"""
        date = self._parse_date(date)[:10]
        curr_value = self.get_value(date)
        return float(curr_value - self.net_deposits)

    def get_value(self, date: DateLike | None = None) -> float:
        """Portfolio market value at date"""
        date = self._parse_date(date)
        return sum(self.holdings_value(date).values()) + self.cash

    def holdings_pnl(self, date: DateLike | None = None) -> dict:
        """PnL in absolute currency of each holdings at date"""
        date = self._parse_date(date)
        date = date.strftime('%Y-%m-%d') if type(date) != str else date[:10]
        value = self.holdings_value(date)
        return {ast: float(value[ast] - (self.holdings[ast] * self.cost_bases[ast]))
                for ast in self.assets}

    def holdings_returns(self, date: DateLike | None = None) -> dict:
        """Returns in decimals of each holdings at date"""
        date = self._parse_date(date)
        date = date.strftime('%Y-%m-%d') if type(date) != str else date[:10]
        pnl = self.holdings_pnl(date)
        return {ast: float(pnl[ast] / (self.holdings[ast] * self.cost_bases[ast]))
                for ast in self.assets}

    def holdings_value(self, date: DateLike | None = None) -> dict:
        """Market value of each holdings at date"""
        date = self._parse_date(date)[:10]

        return {asset: float(self._get_price(asset, date) * shares)
                for asset, shares in self.holdings.items()}
    
    @property
    def ann_factor(self):
        stock_weight = sum(v for k, v in self.weights.items() if k.asset_type != 'Cryptocurrency')
        crypto_weight = sum(v for k, v in self.weights.items() if k.asset_type == 'Cryptocurrency')
        return (stock_weight * 252) + (crypto_weight * 365)

    @property
    def volatility(self):
        if not self.weights:
            return 0

        daily_vol = self.returns.std()
        return daily_vol * np.sqrt(self.ann_factor)

    @property
    def annualized_returns(self):

        return (1 + self.returns.mean()) ** self.ann_factor - 1

    @property
    def downside_deviation(self):
        returns = self.returns
        return np.sqrt(np.mean(np.minimum(returns, 0) ** 2))

    @property
    def sortino_ratio(self):
        if not self.weights:
            return 0

        downside_deviation = self.downside_deviation
        if downside_deviation == 0:
            return 0

        ann_factor = self.ann_factor

        # Convert annual risk-free rate to daily using weighted factor
        daily_rf = self.r / ann_factor

        # Calculate excess returns
        excess_returns = self.returns - daily_rf

        # Annualize mean excess returns
        mean_excess_returns = excess_returns.mean() * ann_factor
        print(f"Mean excess returns (pre-ann): {excess_returns.mean()}")
        print(f"Downside deviation: {downside_deviation}")
        print(f"Annual factor: {ann_factor}")
        print(f"Final annualized excess return: {mean_excess_returns}")

        return mean_excess_returns / (downside_deviation * np.sqrt(ann_factor))

    @property
    def weights(self):
        holdings_value = self.holdings_value()
        return {k: v / sum(holdings_value.values()) for k, v in holdings_value.items()}

    @property
    def sharpe_ratio(self):

        if not self.weights:
            return 0

        ann_factor = self.ann_factor

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

    @property
    def beta(self):
        market = Asset('SPY')
        self._convert_ast(market)
        df = pd.DataFrame(index=pd.date_range(start='2020-01-01', end=pd.Timestamp.today()))
        has_crypto = False
        df['market'] = market.daily['log_rets']
        for ast in self.assets:
            df[ast] = ast.daily['log_rets']
            if ast.asset_type == 'Cryptocurrency':
                has_crypto = True

        df = df.ffill() if has_crypto else df.dropna()

        df = df.dropna().resample('ME').agg('sum')
        df = np.exp(df)

        betas = {}
        market_var = df['market'].var()
        for col in df.columns:
            if str(col) == 'market':
                continue
            beta = df[col].cov(df['market']) / market_var
            betas[col] = float(beta)

        weights = self.weights
        return sum(weights[ast] * betas[ast] for ast in self.assets)

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

    def risk_decomposition(self):
        df = pd.DataFrame()
        port_weights = self.weights
        weights = []
        for ast in self.assets:
            df[ast] = ast.daily['rets']
            weights.append(port_weights[ast])
        weights = np.array(weights)

        cov = df.cov() * self.ann_factor
        port_vol = np.sqrt(weights.T @ cov @ weights)
        marginal_risk = (cov @ weights) / port_vol
        component_risk = marginal_risk * weights
        risk_contribution = component_risk / port_vol * 100

        # Create the data as a dictionary first
        data = {
            'Weight': weights,
            'Risk Contribution': risk_contribution,
            'Marginal Risk': marginal_risk,
            'Component Risk': component_risk
        }

        # Create DataFrame all at once with the assets as index
        risk_decomp = pd.DataFrame(data, index=self.assets)
        risk_decomp = risk_decomp.sort_values('Weight', ascending=False)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Weight vs Risk Contribution', 'Marginal Risk'),
            vertical_spacing=0.2,
            row_heights=[0.7, 0.3]
        )

        # Create a color map for assets
        colors = px.colors.qualitative.Set2
        asset_colors = {ast.ticker: colors[i % len(colors)] for i, ast in enumerate(risk_decomp.index)}

        # Add traces for top subplot (stacked)
        for ast in risk_decomp.index:
            # Add weight bar
            fig.add_trace(
                go.Bar(
                    name=ast.ticker,
                    x=['Portfolio Weight'],
                    y=[risk_decomp.loc[ast, 'Weight'] * 100],
                    marker_color=asset_colors[ast.ticker],
                    width=0.5,
                    hovertemplate="<br>".join([
                        "<b>%{x}</b>",
                        f"Asset: {ast.ticker}",
                        "Weight: %{y:.1f}%",
                        "<extra></extra>"
                    ])
                ),
                row=1, col=1
            )

            # Add risk contribution bar
            fig.add_trace(
                go.Bar(
                    name=ast.ticker,
                    x=['Risk Contribution'],
                    y=[risk_decomp.loc[ast, 'Risk Contribution']],
                    marker_color=asset_colors[ast.ticker],
                    showlegend=False,
                    width=0.5,
                    hovertemplate="<br>".join([
                        "<b>%{x}</b>",
                        f"Asset: {ast.ticker}",
                        "Contribution: %{y:.1f}%",
                        "<extra></extra>"
                    ])
                ),
                row=1, col=1
            )

            # Add marginal risk bar (bottom subplot)
            fig.add_trace(
                go.Bar(
                    name=ast.ticker,
                    x=[ast.ticker],
                    y=[risk_decomp.loc[ast, 'Marginal Risk']],
                    marker_color=asset_colors[ast.ticker],
                    showlegend=False,
                    width=0.5,
                    hovertemplate="<br>".join([
                        f"Asset: {ast.ticker}",
                        "Marginal Risk: %{y:.4f}",
                        "<extra></extra>"
                    ])
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            barmode='stack',
            showlegend=True,
            height=800,
            title_text="Portfolio Risk Analysis"
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Percentage (%)", range=[0, 105], row=1, col=1)
        fig.update_yaxes(title_text="Marginal Risk", row=2, col=1)

        # Update x-axis for bottom subplot
        fig.update_xaxes(title_text="Assets", row=2, col=1)

        fig.show()

        return fig

    @property
    def max_drawdown(self):
        cum_rets = (1 + self.returns).cumprod()
        max_dd = (cum_rets / cum_rets.cummax() - 1).min()
        return float(max_dd)

    @property
    def drawdowns(self):
        cum_rets = (1 + self.returns).cumprod()
        drawdown = (cum_rets / cum_rets.cummax() - 1)
        return drawdown.dropna()

    @property
    def longest_drawdown_duration(self):
        drawdown = self.drawdowns
        drawdown_peaks = drawdown[drawdown == 0]
        end_idx = pd.Series(drawdown_peaks.index.diff()).idxmax()
        longest_end = drawdown_peaks.index[end_idx].date()
        longest_start = drawdown_peaks.index[end_idx - 1].date()
        longest_duration = (longest_end - longest_start).days
        return {'start': longest_start.strftime('%d-%m-%Y'), 'end': longest_end.strftime('%d-%m-%Y'), 'duration': longest_duration}

    @property
    def average_drawdown(self):
        drawdown = self.drawdowns[self.drawdowns < 0]
        return float(drawdown.mean())
    
    @property
    def drawdown_ratio(self):
        return self.max_drawdown / self.average_drawdown

    def time_to_recovery(self, min_duration: int = 3, min_depth: float = 0.05):
        df = self.drawdown_df
        return float(df[(df['duration'] >= min_duration) & (-df['depth'] >= np.abs(min_depth))]['time_to_recovery'].mean())

    def average_drawdown_duration(self, min_duration: int = 3, min_depth: float = 0.05):
        df = self.drawdown_df
        return float(df[(df['duration'] >= min_duration) & (-df['depth'] >= np.abs(min_depth))]['duration'].mean())

    def drawdown_frequency(self, bins: int = 20):
        df = self.drawdown_df
        troughs = df['depth']
        bins = np.linspace(troughs.min(), troughs.max(), bins + 1)

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
            x=troughs, 
            xbins=dict(
                start=bins[0], 
                end=bins[-1], 
                size=(bins[1] - bins[0])
                ),
            name='Drawdown Frequency'
            )
        )

        fig.update_layout(
            title='Drawdown Frequency',
            xaxis_title='Drawdown Depth',
            yaxis_title='Frequency',
            yaxis=dict(
                    range=[0, None],
                    rangemode='nonnegative'
                ),
            bargap=0.05
        )
        fig.show()

        return troughs

    @property
    def drawdown_df(self):
        drawdown_series = self.drawdowns
        # Initialize variables
        in_drawdown = False
        drawdown_start = None
        bottom_date = None
        current_bottom = 0
        recovery_periods = []

        for date, dd in drawdown_series.items():
            # New drawdown starts
            if not in_drawdown and dd < 0:
                in_drawdown = True
                drawdown_start = date
                bottom_date = date
                current_bottom = dd

            # During drawdown
            elif in_drawdown:
                # Update bottom if we go lower
                if dd < current_bottom:
                    bottom_date = date
                    current_bottom = dd

                # Recovery found
                if dd == 0:
                    recovery_date = date
                    in_drawdown = False
                    recovery_periods.append({
                        'start': drawdown_start,
                        'bottom': bottom_date,
                        'recovery': recovery_date,
                        'depth': current_bottom,
                        'time_to_recovery': (recovery_date - bottom_date).days,
                        'duration': (recovery_date - drawdown_start).days
                    })

        # If still in drawdown at end of series
        if in_drawdown:
            recovery_periods.append({
                'start': drawdown_start,
                'bottom': bottom_date,
                'recovery': None,
                'depth': current_bottom,
                'time_to_recovery': None,
                'duration': (date - drawdown_start).days
            })

        return pd.DataFrame(recovery_periods)

    @property
    def calmar_ratio(self):
        return float(self.annualized_returns / np.abs(self.max_drawdown))

    @property
    def drawdown_metrics(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.drawdowns.index,
                y=self.drawdowns,
                mode='lines',
                name='Drawdown'
            )
        )

        fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            showlegend=False
        )

        fig.show()

        metrics = {
            'max_drawdown': self.max_drawdown,
            'average_drawdown': self.average_drawdown,
            'drawdown_ratio': self.drawdown_ratio,
            'longest_drawdown_duration': self.longest_drawdown_duration,
            'time_to_recovery': self.time_to_recovery(),
            'average_drawdown_duration': self.average_drawdown_duration(),
            'calmar_ratio': self.calmar_ratio
        }

        return metrics

    def save(self, name):
        pass

    @classmethod
    def load(cls, name):
        pass

    @classmethod
    def report(cls, name):
        pass
