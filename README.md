# Financial Analysis Library

A Python library for financial data analysis and visualization. The library integrates with a PostgreSQL database for data storage while automatically handling missing data through the yfinance API. Designed for both quick analysis and integration into larger projects.

## Features

- **Simple API with Automatic Data Management**
  - Easy-to-use Asset class for financial analysis
  - Automatic download and storage of missing data
  - Handles new assets and currencies seamlessly

- **Comprehensive Visualization**
  - Price history plots
  - Candlestick charts with volume
  - Returns distribution analysis
  - Technical indicators (SMA, Bollinger Bands)

- **Flexible Plot Creation**
  - Both static (matplotlib) and interactive (plotly) visualizations
  - Support for standalone plots and subplots
  - Customizable formatting through returned figure objects

- **Technical Analysis Tools**
  - Simple Moving Averages (SMA)
  - Exponential Moving Averages (EMA)
  - Bollinger Bands
  - SMA Crossover strategy

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

3. Ensure you have the [financial database](https://github.com/ZMCodi/stock-db) set up

## Quick Start

```python
from assets import Asset
import plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Initialize assets
# Note: If these tickers aren't in the database, they will be automatically 
# downloaded from yfinance and added to the database
aapl = Asset('AAPL')
msft = Asset('MSFT')
nvda = Asset('NVDA')

# Interactive Plotting Example
# Create subplots with different chart types
fig1 = make_subplots(rows=3, cols=1)
aapl.plot_price_history(interactive=True, fig=fig1, subplot_idx=(1, 1))
msft.plot_candlestick(interactive=True, fig=fig1, vol_idx=(3, 1), candle_idx=(2, 1))
fig1.show()

# Create standalone interactive plot
nvda.SMA_crossover(interactive=True, short=20, long=150)

# Static Plotting Example
# Create subplot figure with different analysis
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
aapl.plot_price_history(fig=fig2, subplot_idx=0)
nvda.plot_returns_dist(ax=ax2)
msft.plot_candlestick(fig=fig2, volume=False, candle_idx=2)
plt.show()

# Create standalone static plot with technical indicators
aapl.plot_SMA(alpha=0.3, bollinger_bands=True)
```

## API Overview

### Data Management
- `Asset(ticker)`: Initialize an asset and load its data
- `get_data()`: Retrieve asset data from database
- `insert_new_ticker()`: Add new asset to database
- `resample(period)`: Resample data to different timeframes

### Basic Analysis
- `basic_stats()`: Get key statistics about the asset
- `rolling_stats()`: Calculate rolling statistics
- `add_bollinger_bands()`: Add Bollinger Bands to analysis

### Visualization
#### Price Charts
- `plot_price_history()`: Plot price over time
- `plot_candlestick()`: Create candlestick chart with optional volume
- `plot_returns_dist()`: Visualize returns distribution

#### Technical Analysis
- `plot_SMA()`: Plot Simple Moving Average with optional Bollinger Bands
- `SMA_crossover()`: Plot SMA crossover signals

### Common Parameters
Most visualization methods support:
- `interactive`: Toggle between static (matplotlib) and interactive (plotly) plots
- `start_date`/`end_date`: Limit data range
- `timeframe`: Choose between daily ('1d') and 5-minute data
- `fig`/`subplot_idx`: Integration with existing figures for subplots
- `filename`: Save plot to file

## Future Development

- Backtest implementation for SMA strategy
- More statistical analysis metrics
- SMA window optimization
- Additional technical indicators and plot types
- Default dashboard implementation
- Portfolio class integration for multi-asset analysis

## Notes

- All plots return figure objects for further customization
- Static plots use matplotlib/seaborn while interactive plots use plotly
- Data is automatically managed through database integration with fallback to yfinance
- Supports both daily and 5-minute data timeframes