# Financial Analysis and Trading Strategy Library

A Python library for financial data analysis and trading strategy implementation. The library integrates with a PostgreSQL database for data storage while automatically handling missing data through the yfinance API. It provides both analytical tools and a comprehensive framework for developing, testing, and optimizing trading strategies.

## Features

- **Simple API with Automatic Data Management**
  - Easy-to-use Asset class for financial analysis
  - Automatic download and storage of missing data
  - Handles new assets and currencies seamlessly

- **Comprehensive Visualization**
  - Price history plots
  - Candlestick charts with volume
  - Returns distribution analysis
  - Technical indicators with signals
  - Strategy performance visualization

- **Technical Analysis and Trading Strategies**
  - Moving Average Crossover
  - RSI with multiple signal types
  - MACD with momentum and divergence signals
  - Bollinger Bands with squeeze and breakout detection
  - Strategy combination and signal weighting

- **Trading Framework**
  - Strategy backtesting with daily and 5-minute data
  - Parameter optimization through grid search
  - Signal generation and combination methods
  - Weight optimization for combined strategies
  - Performance analysis and visualization
  - Parallel backtesting and optimization

- **Flexible Plot Creation**
  - Both static (matplotlib) and interactive (plotly) visualizations
  - Support for standalone plots and subplots
  - Customizable formatting through returned figure objects
  - Strategy-specific visualization methods

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ZMCodi/assets.git
cd assets
```

2. Create and activate the Python environment:
```bash
conda env create -f environment.yml
conda activate [environment-name]
```

3. It is recommended that you have the [financial database](https://github.com/ZMCodi/stock-db) set up. But you can still use the Asset class without the database. Just pass in `from_db=False` when instantiating the Asset object.

## Quick Start

```python
from assets import Asset
from strategy import MA_Crossover, RSI, MACD, BB, CombinedStrategy

# Initialize asset
aapl = Asset('AAPL')
# or if you don't have the database
aapl = Asset('AAPL', from_db=False)

# Basic Analysis
aapl.plot_price_history(interactive=True)
aapl.plot_candlestick(volume=True)
aapl.plot_returns_dist(show_stats=True)

# Simple Moving Average Strategy
ma_strat = MA_Crossover(aapl, short_window=20, long_window=50)
ma_strat.plot()  # View strategy signals
results = ma_strat.backtest()  # Run backtest
ma_strat.optimize(inplace=True)  # Optimize parameters

# RSI Strategy with Multiple Signals
rsi_strat = RSI(aapl, 
                signal_type=['crossover', 'divergence', 'hidden divergence'],
                combine='weighted')
rsi_strat.plot(candlestick=True)
rsi_strat.optimize_weights(inplace=True)  # Optimize signal weights

# Combined Strategy Approach
strategies = [
    MA_Crossover(aapl, short_window=20, long_window=50),
    RSI(aapl, ub=70, lb=30),
    MACD(aapl),
    BB(aapl)
]
combined = CombinedStrategy(aapl, strategies=strategies)
combined.backtest()
combined.optimize_weights(inplace=True)  # Optimize strategy weights

```

\**Check demo.ipynb for more detailed usage*

## API Overview

### Asset Management
- `Asset(ticker)`: Initialize an asset and load its data
- `get_data()`: Retrieve asset data from database
- `resample(period)`: Resample data to different timeframes

### Analysis Tools
- `stats`: Get key statistics and metrics
- `rolling_stats()`: Calculate rolling window statistics
- Technical indicators calculation through TAEngine class

### Technical Analysis Strategies
- `MA_Crossover`: Moving average crossover strategy
  - Traditional crossover signals
  - Parameter optimization
  - Performance visualization

- `RSI`: Relative Strength Index strategy
  - Multiple signal types (crossover, divergence)
  - Mean reversion capability
  - Weight optimization

- `MACD`: Moving Average Convergence Divergence
  - Multiple signal types
  - Momentum analysis
  - Double peak/trough detection

- `BB`: Bollinger Bands
  - Multiple signal types (bounce, squeeze, breakout)
  - Volatility-based signals
  - Band walk detection

- `CombinedStrategy`: Strategy integration
  - Multiple strategy combination
  - Weight optimization
  - Consensus-based signals

### Strategy Framework
- `backtest()`: Run strategy backtest
- `optimize()`: Grid search parameter optimization
- `optimize_weights()`: Signal weight optimization
- `plot()`: Strategy-specific visualization
- Signal generation and combination utilities

### Visualization
- `plot_price_history()`: Plot price over time
- `plot_candlestick()`: Create candlestick chart with volume
- `plot_returns_dist()`: Visualize returns distribution
- Strategy plotting methods with signals

### Common Parameters
Most visualization and strategy methods support:
- `interactive`: Toggle between static and interactive plots
- `timeframe`: Choose between daily ('1d') and 5-minute data
- `start_date`/`end_date`: Limit data range
- `fig`/`subplot_idx`: Integration with existing figures
- `candlestick`: Toggle between line and candlestick charts
- `show_signal`: Display trading signals

## Notes

- All plots return figure objects for further customization
- Static plots use matplotlib/seaborn while interactive plots use plotly
- Data is automatically managed through database integration with fallback to yfinance
- Supports both daily and 5-minute data timeframes
- Strategy optimization supports multiple approaches and objectives
