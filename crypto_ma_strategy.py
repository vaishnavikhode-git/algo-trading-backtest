import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MovingAverageCrossoverStrategy:
    """
    A class to implement and backtest a moving average crossover strategy
    for cryptocurrency trading simulation.
    """
    
    def __init__(self, short_window=5, long_window=20, initial_capital=10000):
        """
        Initialize the strategy parameters.
        
        Parameters:
        -----------
        short_window : int
            Period for short-term moving average (default: 5)
        long_window : int
            Period for long-term moving average (default: 20)
        initial_capital : float
            Starting capital for backtesting (default: 10000)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.positions = None
        self.portfolio = None
        
    def generate_mock_data(self, start_date='2023-01-01', end_date='2024-01-01', 
                          initial_price=30000, volatility=0.02):
        """
        Generate mock Bitcoin price data using geometric Brownian motion.
        
        Parameters:
        -----------
        start_date : str
            Start date for the data (YYYY-MM-DD format)
        end_date : str
            End date for the data (YYYY-MM-DD format)
        initial_price : float
            Starting price for BTC (default: 30000)
        volatility : float
            Daily volatility (default: 0.02 = 2%)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with datetime index and BTC price data
        """
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Generate random returns using geometric Brownian motion
        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(0.001, volatility, n_days)  # Small positive drift
        
        # Calculate cumulative prices
        prices = [initial_price]
        for i in range(1, n_days):
            price = prices[-1] * (1 + daily_returns[i])
            prices.append(price)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'price': prices,
            'volume': np.random.normal(1000000, 200000, n_days)  # Mock volume data
        }, index=dates)
        
        return self.data
    
    def calculate_signals(self):
        """
        Calculate moving averages and generate buy/sell signals.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with price data, moving averages, and signals
        """
        if self.data is None:
            raise ValueError("No data available. Please generate or load data first.")
        
        # Calculate moving averages
        self.data[f'MA_{self.short_window}'] = self.data['price'].rolling(
            window=self.short_window, min_periods=1).mean()
        self.data[f'MA_{self.long_window}'] = self.data['price'].rolling(
            window=self.long_window, min_periods=1).mean()
        
        # Generate signals (1 for buy, -1 for sell, 0 for hold)
        self.data['signal'] = 0
        self.data['signal'][self.short_window:] = np.where(
            self.data[f'MA_{self.short_window}'][self.short_window:] > 
            self.data[f'MA_{self.long_window}'][self.short_window:], 1, -1)
        
        # Generate trading signals (only when signal changes)
        self.data['positions'] = self.data['signal'].diff()
        
        return self.data
    
    def backtest_strategy(self):
        """
        Backtest the moving average crossover strategy.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio performance metrics
        """
        if self.data is None or 'signal' not in self.data.columns:
            raise ValueError("Signals not calculated. Please run calculate_signals() first.")
        
        # Initialize portfolio
        self.portfolio = pd.DataFrame(index=self.data.index)
        self.portfolio['price'] = self.data['price']
        self.portfolio['signal'] = self.data['signal']
        self.portfolio['positions'] = self.data['positions']
        
        # Calculate daily returns
        self.portfolio['market_returns'] = self.data['price'].pct_change()
        
        # Calculate strategy returns (only when holding position)
        self.portfolio['strategy_returns'] = (
            self.portfolio['market_returns'] * self.portfolio['signal'].shift(1)
        )
        
        # Calculate cumulative returns
        self.portfolio['cumulative_market_returns'] = (
            1 + self.portfolio['market_returns']).cumprod()
        self.portfolio['cumulative_strategy_returns'] = (
            1 + self.portfolio['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        self.portfolio['portfolio_value'] = (
            self.initial_capital * self.portfolio['cumulative_strategy_returns']
        )
        
        return self.portfolio
    
    def calculate_performance_metrics(self):
        """
        Calculate key performance metrics for the strategy.
        
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if self.portfolio is None:
            raise ValueError("Portfolio not calculated. Please run backtest_strategy() first.")
        
        # Basic metrics
        total_trades = len(self.portfolio[self.portfolio['positions'] != 0])
        buy_signals = len(self.portfolio[self.portfolio['positions'] == 2])  # Signal change from -1 to 1
        sell_signals = len(self.portfolio[self.portfolio['positions'] == -2])  # Signal change from 1 to -1
        
        # Returns
        total_strategy_return = self.portfolio['cumulative_strategy_returns'].iloc[-1] - 1
        total_market_return = self.portfolio['cumulative_market_returns'].iloc[-1] - 1
        
        # Risk metrics
        strategy_volatility = self.portfolio['strategy_returns'].std() * np.sqrt(252)
        market_volatility = self.portfolio['market_returns'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (
            self.portfolio['strategy_returns'].mean() / 
            self.portfolio['strategy_returns'].std() * np.sqrt(252)
        ) if self.portfolio['strategy_returns'].std() != 0 else 0
        
        # Maximum drawdown
        rolling_max = self.portfolio['cumulative_strategy_returns'].expanding().max()
        drawdown = (self.portfolio['cumulative_strategy_returns'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'Total Trades': total_trades,
            'Buy Signals': buy_signals,
            'Sell Signals': sell_signals,
            'Strategy Return': f"{total_strategy_return:.2%}",
            'Market Return': f"{total_market_return:.2%}",
            'Strategy Volatility': f"{strategy_volatility:.2%}",
            'Market Volatility': f"{market_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Final Portfolio Value': f"${self.portfolio['portfolio_value'].iloc[-1]:,.2f}"
        }
        
        return metrics
    
    def plot_strategy(self, figsize=(15, 10)):
        """
        Plot the strategy results including price, moving averages, and signals.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        if self.portfolio is None:
            raise ValueError("Portfolio not calculated. Please run backtest_strategy() first.")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Price and Moving Averages with Buy/Sell Signals
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['price'], label='BTC Price', linewidth=1, color='black')
        ax1.plot(self.data.index, self.data[f'MA_{self.short_window}'], 
                label=f'{self.short_window}-Day MA', linewidth=1, color='blue')
        ax1.plot(self.data.index, self.data[f'MA_{self.long_window}'], 
                label=f'{self.long_window}-Day MA', linewidth=1, color='red')
        
        # Add buy/sell signals
        buy_signals = self.portfolio[self.portfolio['positions'] == 2]
        sell_signals = self.portfolio[self.portfolio['positions'] == -2]
        
        ax1.scatter(buy_signals.index, buy_signals['price'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['price'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Bitcoin Price with Moving Average Crossover Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Performance
        ax2 = axes[1]
        ax2.plot(self.portfolio.index, 
                self.portfolio['cumulative_strategy_returns'], 
                label='Strategy Returns', linewidth=2, color='green')
        ax2.plot(self.portfolio.index, 
                self.portfolio['cumulative_market_returns'], 
                label='Buy & Hold Returns', linewidth=2, color='blue')
        
        ax2.set_title('Strategy Performance vs Buy & Hold', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Returns', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline: generate data, calculate signals, 
        backtest, and display results.
        """
        print("🚀 Running Moving Average Crossover Strategy Analysis")
        print("=" * 60)
        
        # Generate mock data
        print("📊 Generating mock Bitcoin price data...")
        self.generate_mock_data()
        print(f"✅ Generated {len(self.data)} days of price data")
        
        # Calculate signals
        print("\n📈 Calculating moving averages and signals...")
        self.calculate_signals()
        
        # Backtest strategy
        print("🔄 Backtesting strategy...")
        self.backtest_strategy()
        
        # Calculate performance metrics
        print("\n📋 Performance Metrics:")
        print("-" * 30)
        metrics = self.calculate_performance_metrics()
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Plot results
        print("\n📊 Generating strategy visualization...")
        self.plot_strategy()
        
        return metrics


def main():
    """
    Main function to demonstrate the moving average crossover strategy.
    """
    # Initialize strategy
    strategy = MovingAverageCrossoverStrategy(
        short_window=5, 
        long_window=20, 
        initial_capital=10000
    )
    
    # Run complete analysis
    strategy.run_complete_analysis()


if __name__ == "__main__":
    main()
