import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import time

# Import all strategy modules
from strategies.momentum_strategy import MomentumStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.volume_profile_strategy import VolumeProfileStrategy
from strategies.pairs_trading_strategy import PairsTradingStrategy
from strategies.grid_trading_strategy import GridTradingStrategy

# Import utilities
from utils.risk_manager import create_risk_manager
from utils.portfolio_manager import create_portfolio_manager
from utils.performance_tracker import create_performance_tracker
from utils.market_scanner import create_market_scanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyTester:
    """Comprehensive Strategy Testing Framework"""
    
    def __init__(self):
        self.strategies = {}
        self.test_results = {}
        self.market_data = {}
        self.performance_tracker = None
        
        # Test configuration
        self.test_config = {
            'initial_capital': 10000,
            'test_period_days': 30,
            'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT'],
            'commission_rate': 0.001,  # 0.1% commission
            'slippage_rate': 0.0005,   # 0.05% slippage
            'max_positions': 5
        }
        
        # Mock configurations
        self.api_config = self._create_mock_api_config()
        self.trading_config = self._create_mock_trading_config()
        
    def _create_mock_api_config(self):
        """Create mock API configuration"""
        class MockApiConfig:
            def __init__(self):
                self.binance_api_key = "test_key"
                self.binance_secret = "test_secret"
                self.binance_testnet = True
        
        return MockApiConfig()
    
    def _create_mock_trading_config(self):
        """Create mock trading configuration"""
        class MockTradingConfig:
            def __init__(self):
                self.initial_capital = 10000
                self.max_open_positions = 5
                self.max_risk_per_trade = 0.02
                self.daily_loss_limit = 0.05
                self.max_consecutive_losses = 3
        
        return MockTradingConfig()
    
    async def initialize(self):
        """Initialize testing framework"""
        try:
            logger.info("Initializing Strategy Testing Framework...")
            
            # Initialize performance tracker
            self.performance_tracker = create_performance_tracker(
                initial_capital=self.test_config['initial_capital']
            )
            await self.performance_tracker.initialize()
            
            # Generate mock market data
            await self._generate_mock_market_data()
            
            # Initialize strategies
            await self._initialize_strategies()
            
            logger.info("Strategy Testing Framework initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize testing framework: {e}")
            raise
    
    async def _generate_mock_market_data(self):
        """Generate realistic mock market data for testing"""
        try:
            logger.info("Generating mock market data...")
            
            for symbol in self.test_config['symbols']:
                # Generate OHLCV data
                periods = self.test_config['test_period_days'] * 24  # Hourly data
                dates = pd.date_range(
                    start=datetime.utcnow() - timedelta(days=self.test_config['test_period_days']),
                    periods=periods,
                    freq='H'
                )
                
                # Set base prices for different symbols
                base_prices = {
                    'BTC/USDT': 45000,
                    'ETH/USDT': 3000,
                    'ADA/USDT': 0.45,
                    'DOT/USDT': 25.0
                }
                
                base_price = base_prices.get(symbol, 1000)
                
                # Generate realistic price data with trends and volatility
                np.random.seed(42)  # For reproducible results
                
                prices = [base_price]
                volumes = []
                
                # Create some market trends
                trend_changes = np.random.choice([0, 1, 2], periods, p=[0.7, 0.15, 0.15])
                # 0 = sideways, 1 = uptrend, 2 = downtrend
                
                for i in range(1, periods):
                    # Base volatility
                    volatility = 0.02 if symbol.startswith('BTC') or symbol.startswith('ETH') else 0.03
                    
                    # Trend component
                    trend = trend_changes[i]
                    if trend == 1:  # Uptrend
                        drift = np.random.normal(0.0005, 0.001)
                    elif trend == 2:  # Downtrend
                        drift = np.random.normal(-0.0005, 0.001)
                    else:  # Sideways
                        drift = np.random.normal(0, 0.0002)
                    
                    # Random component
                    random_change = np.random.normal(0, volatility)
                    
                    # Calculate new price
                    price_change = drift + random_change
                    new_price = prices[-1] * (1 + price_change)
                    
                    # Prevent unrealistic price movements
                    new_price = max(new_price, base_price * 0.5)
                    new_price = min(new_price, base_price * 2.0)
                    
                    prices.append(new_price)
                    
                    # Generate volume (with occasional spikes)
                    base_volume = np.random.lognormal(15, 0.5)
                    if np.random.random() < 0.05:  # 5% chance of volume spike
                        base_volume *= np.random.uniform(2, 5)
                    volumes.append(base_volume)
                
                # Create OHLC from price data
                ohlc_data = []
                for i in range(len(prices)):
                    if i == 0:
                        open_price = prices[i]
                        close_price = prices[i]
                        high_price = prices[i] * 1.01
                        low_price = prices[i] * 0.99
                        volume = volumes[0] if volumes else 1000000
                    else:
                        open_price = prices[i-1]
                        close_price = prices[i]
                        
                        # Generate realistic high/low
                        price_range = abs(close_price - open_price)
                        high_price = max(open_price, close_price) + price_range * np.random.uniform(0, 0.5)
                        low_price = min(open_price, close_price) - price_range * np.random.uniform(0, 0.5)
                        
                        volume = volumes[i-1] if i-1 < len(volumes) else 1000000
                    
                    ohlc_data.append([
                        dates[i],
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume
                    ])
                
                # Create DataFrame
                df = pd.DataFrame(ohlc_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                
                self.market_data[symbol] = df
                
                logger.info(f"Generated {len(df)} data points for {symbol} "
                          f"(${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f})")
            
        except Exception as e:
            logger.error(f"Error generating mock market data: {e}")
            raise
    
    async def _initialize_strategies(self):
        """Initialize all strategies for testing"""
        try:
            logger.info("Initializing strategies...")
            
            # Momentum Strategy
            momentum_config = {
                'lookback_period': 14,
                'momentum_threshold': 0.02,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06,
                'volume_filter': True,
                'min_volume_ratio': 1.5
            }
            
            momentum_strategy = MomentumStrategy(momentum_config)
            await momentum_strategy.initialize()
            self.strategies['momentum'] = momentum_strategy
            
            # RSI Strategy
            rsi_config = {
                'rsi_period': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05,
                'volume_confirmation': True
            }
            
            rsi_strategy = RSIStrategy(rsi_config)
            await rsi_strategy.initialize()
            self.strategies['rsi'] = rsi_strategy
            
            # Volume Profile Strategy
            volume_config = {
                'profile_period': 24,
                'value_area_percentage': 0.7,
                'poc_threshold': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
            
            volume_strategy = VolumeProfileStrategy(volume_config)
            await volume_strategy.initialize()
            self.strategies['volume_profile'] = volume_strategy
            
            # Pairs Trading Strategy
            pairs_config = {
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.0,
                'correlation_threshold': 0.7
            }
            
            pairs_strategy = PairsTradingStrategy(pairs_config)
            await pairs_strategy.initialize()
            self.strategies['pairs_trading'] = pairs_strategy
            
            # Grid Trading Strategy
            grid_config = {
                'grid_size': 0.02,
                'num_grids': 10,
                'base_position_size': 0.1,
                'take_profit_pct': 0.015,
                'stop_loss_pct': 0.05
            }
            
            grid_strategy = GridTradingStrategy(grid_config)
            await grid_strategy.initialize()
            self.strategies['grid_trading'] = grid_strategy
            
            logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            raise
    
    async def test_single_strategy(self, strategy_name: str, symbol: str) -> Dict[str, Any]:
        """Test a single strategy on a single symbol"""
        try:
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy {strategy_name} not found")
            
            if symbol not in self.market_data:
                raise ValueError(f"No market data for {symbol}")
            
            strategy = self.strategies[strategy_name]
            market_data = self.market_data[symbol]
            
            logger.info(f"Testing {strategy_name} strategy on {symbol}...")
            
            # Test results tracking
            signals = []
            trades = []
            portfolio_value = self.test_config['initial_capital']
            cash_balance = portfolio_value
            positions = {}
            total_pnl = 0
            
            # Process each data point
            for i in range(len(market_data)):
                current_data = market_data.iloc[:i+1]
                
                if len(current_data) < 20:  # Need minimum data for indicators
                    continue
                
                current_price = current_data['close'].iloc[-1]
                current_time = current_data['timestamp'].iloc[-1]
                
                # Generate signal
                try:
                    signal = await strategy.generate_signal(symbol, current_data, current_price)
                    
                    if signal and signal.signal_type != 'HOLD':
                        signals.append({
                            'timestamp': current_time,
                            'signal_type': signal.signal_type,
                            'confidence': signal.confidence,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'reasoning': signal.reasoning
                        })
                        
                        # Simulate trade execution
                        if signal.signal_type in ['BUY', 'LONG'] and symbol not in positions:
                            # Calculate position size (risk-based)
                            risk_amount = portfolio_value * 0.02  # 2% risk per trade
                            stop_distance = abs(signal.entry_price - signal.stop_loss) if signal.stop_loss else signal.entry_price * 0.03
                            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                            
                            # Limit position size to available cash
                            max_position_value = cash_balance * 0.9  # Leave some cash
                            max_position_size = max_position_value / signal.entry_price
                            position_size = min(position_size, max_position_size)
                            
                            if position_size > 0:
                                position_value = position_size * signal.entry_price
                                commission = position_value * self.test_config['commission_rate']
                                slippage = position_value * self.test_config['slippage_rate']
                                total_cost = position_value + commission + slippage
                                
                                if cash_balance >= total_cost:
                                    positions[symbol] = {
                                        'size': position_size,
                                        'entry_price': signal.entry_price,
                                        'entry_time': current_time,
                                        'stop_loss': signal.stop_loss,
                                        'take_profit': signal.take_profit,
                                        'strategy': strategy_name
                                    }
                                    cash_balance -= total_cost
                        
                        elif signal.signal_type in ['SELL', 'SHORT'] and symbol in positions:
                            # Close position
                            position = positions[symbol]
                            exit_price = signal.entry_price  # Use signal price as exit price
                            
                            pnl = (exit_price - position['entry_price']) * position['size']
                            commission = position['size'] * exit_price * self.test_config['commission_rate']
                            slippage = position['size'] * exit_price * self.test_config['slippage_rate']
                            net_pnl = pnl - commission - slippage
                            
                            # Record trade
                            trade = {
                                'symbol': symbol,
                                'strategy': strategy_name,
                                'direction': 'LONG',
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'size': position['size'],
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'pnl': net_pnl,
                                'pnl_pct': (net_pnl / (position['size'] * position['entry_price'])) * 100,
                                'hold_duration': (current_time - position['entry_time']).total_seconds() / 3600,
                                'exit_reason': 'SIGNAL'
                            }
                            
                            trades.append(trade)
                            cash_balance += position['size'] * exit_price - commission - slippage
                            total_pnl += net_pnl
                            del positions[symbol]
                
                except Exception as e:
                    logger.warning(f"Error generating signal for {symbol} at {current_time}: {e}")
                    continue
                
                # Check stop loss and take profit
                positions_to_close = []
                for pos_symbol, position in positions.items():
                    if pos_symbol == symbol:
                        hit_stop = False
                        hit_target = False
                        exit_reason = None
                        exit_price = current_price
                        
                        if position['stop_loss'] and current_price <= position['stop_loss']:
                            hit_stop = True
                            exit_reason = 'STOP_LOSS'
                            exit_price = position['stop_loss']
                        elif position['take_profit'] and current_price >= position['take_profit']:
                            hit_target = True
                            exit_reason = 'TAKE_PROFIT'
                            exit_price = position['take_profit']
                        
                        if hit_stop or hit_target:
                            pnl = (exit_price - position['entry_price']) * position['size']
                            commission = position['size'] * exit_price * self.test_config['commission_rate']
                            slippage = position['size'] * exit_price * self.test_config['slippage_rate']
                            net_pnl = pnl - commission - slippage
                            
                            trade = {
                                'symbol': symbol,
                                'strategy': strategy_name,
                                'direction': 'LONG',
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'size': position['size'],
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'pnl': net_pnl,
                                'pnl_pct': (net_pnl / (position['size'] * position['entry_price'])) * 100,
                                'hold_duration': (current_time - position['entry_time']).total_seconds() / 3600,
                                'exit_reason': exit_reason
                            }
                            
                            trades.append(trade)
                            cash_balance += position['size'] * exit_price - commission - slippage
                            total_pnl += net_pnl
                            positions_to_close.append(pos_symbol)
                
                # Remove closed positions
                for pos_symbol in positions_to_close:
                    if pos_symbol in positions:
                        del positions[pos_symbol]
                
                # Update portfolio value
                position_value = sum(pos['size'] * current_price for pos in positions.values())
                portfolio_value = cash_balance + position_value
            
            # Close any remaining positions at final price
            final_price = market_data['close'].iloc[-1]
            final_time = market_data['timestamp'].iloc[-1]
            
            for pos_symbol, position in positions.items():
                pnl = (final_price - position['entry_price']) * position['size']
                commission = position['size'] * final_price * self.test_config['commission_rate']
                net_pnl = pnl - commission
                
                trade = {
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'direction': 'LONG',
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'size': position['size'],
                    'entry_time': position['entry_time'],
                    'exit_time': final_time,
                    'pnl': net_pnl,
                    'pnl_pct': (net_pnl / (position['size'] * position['entry_price'])) * 100,
                    'hold_duration': (final_time - position['entry_time']).total_seconds() / 3600,
                    'exit_reason': 'END_OF_TEST'
                }
                
                trades.append(trade)
                total_pnl += net_pnl
            
            # Calculate final portfolio value
            final_portfolio_value = self.test_config['initial_capital'] + total_pnl
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(trades, final_portfolio_value)
            
            results = {
                'strategy': strategy_name,
                'symbol': symbol,
                'test_period': f"{self.test_config['test_period_days']} days",
                'initial_capital': self.test_config['initial_capital'],
                'final_value': final_portfolio_value,
                'total_return': ((final_portfolio_value - self.test_config['initial_capital']) / self.test_config['initial_capital']) * 100,
                'total_pnl': total_pnl,
                'total_signals': len(signals),
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t['pnl'] > 0]),
                'losing_trades': len([t for t in trades if t['pnl'] < 0]),
                'win_rate': (len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100) if trades else 0,
                'avg_trade_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
                'best_trade': max(trades, key=lambda x: x['pnl'])['pnl'] if trades else 0,
                'worst_trade': min(trades, key=lambda x: x['pnl'])['pnl'] if trades else 0,
                'signals': signals,
                'trades': trades,
                'performance_metrics': performance_metrics
            }
            
            logger.info(f"Completed testing {strategy_name} on {symbol}: "
                      f"{results['total_return']:+.2f}% return, {results['total_trades']} trades")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing strategy {strategy_name} on {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, trades: List[Dict], final_value: float) -> Dict[str, Any]:
        """Calculate performance metrics from trades"""
        try:
            if not trades:
                return {}
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = (len(winning_trades) / total_trades) * 100
            
            # P&L metrics
            total_pnl = sum(t['pnl'] for t in trades)
            gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average metrics
            avg_winning_trade = gross_profit / len(winning_trades) if winning_trades else 0
            avg_losing_trade = -gross_loss / len(losing_trades) if losing_trades else 0
            
            # Hold time metrics
            hold_times = [t['hold_duration'] for t in trades]
            avg_hold_time = np.mean(hold_times)
            
            # Expectancy
            expectancy = (win_rate/100 * avg_winning_trade) - ((100-win_rate)/100 * abs(avg_losing_trade))
            
            # Consecutive trades analysis
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            current_win_streak = 0
            current_loss_streak = 0
            
            for trade in trades:
                if trade['pnl'] > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
                else:
                    current_loss_streak += 1
                    current_win_streak = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
            
            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade,
                'best_trade': max(trades, key=lambda x: x['pnl'])['pnl'],
                'worst_trade': min(trades, key=lambda x: x['pnl'])['pnl'],
                'avg_hold_time': avg_hold_time,
                'expectancy': expectancy,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'sharpe_ratio': self._calculate_sharpe_ratio(trades),
                'max_drawdown': self._calculate_max_drawdown(trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trades"""
        try:
            if len(trades) < 2:
                return 0
            
            returns = [t['pnl_pct'] / 100 for t in trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0
            
            # Annualized Sharpe ratio (assuming trades are roughly daily)
            sharpe_ratio = (mean_return * np.sqrt(252)) / (std_return * np.sqrt(252))
            return sharpe_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trades"""
        try:
            if not trades:
                return 0
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = 0
            
            for trade in trades:
                running_total += trade['pnl']
                cumulative_pnl.append(running_total)
            
            # Calculate drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                
                drawdown = (peak - pnl) / abs(peak) if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown * 100  # Return as percentage
            
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {e}")
            return 0
    
    async def test_all_strategies(self) -> Dict[str, Any]:
        """Test all strategies on all symbols"""
        try:
            logger.info("Testing all strategies on all symbols...")
            
            all_results = {}
            strategy_summaries = {}
            
            for strategy_name in self.strategies.keys():
                strategy_results = {}
                strategy_trades = []
                strategy_total_pnl = 0
                
                for symbol in self.test_config['symbols']:
                    result = await self.test_single_strategy(strategy_name, symbol)
                    strategy_results[symbol] = result
                    
                    if 'trades' in result:
                        strategy_trades.extend(result['trades'])
                        strategy_total_pnl += result.get('total_pnl', 0)
                
                # Calculate strategy summary
                if strategy_trades:
                    strategy_performance = self._calculate_performance_metrics(
                        strategy_trades, 
                        self.test_config['initial_capital'] + strategy_total_pnl
                    )
                    
                    strategy_summaries[strategy_name] = {
                        'total_symbols_tested': len(self.test_config['symbols']),
                        'total_trades': len(strategy_trades),
                        'total_pnl': strategy_total_pnl,
                        'total_return_pct': (strategy_total_pnl / self.test_config['initial_capital']) * 100,
                        'performance_metrics': strategy_performance,
                        'best_symbol': max(strategy_results.items(), 
                                         key=lambda x: x[1].get('total_return', -999))[0] if strategy_results else None,
                        'worst_symbol': min(strategy_results.items(), 
                                          key=lambda x: x[1].get('total_return', 999))[0] if strategy_results else None
                    }
                
                all_results[strategy_name] = {
                    'summary': strategy_summaries.get(strategy_name, {}),
                    'detailed_results': strategy_results
                }
            
            # Overall comparison
            comparison = self._compare_strategies(strategy_summaries)
            
            return {
                'test_configuration': self.test_config,
                'test_period': f"{self.test_config['test_period_days']} days",
                'symbols_tested': self.test_config['symbols'],
                'strategies_tested': list(self.strategies.keys()),
                'individual_results': all_results,
                'strategy_comparison': comparison,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error testing all strategies: {e}")
            return {'error': str(e)}
    
    def _compare_strategies(self, strategy_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Compare strategies and rank them"""
        try:
            if not strategy_summaries:
                return {}
            
            # Create ranking table
            ranking_data = []
            
            for strategy_name, summary in strategy_summaries.items():
                performance = summary.get('performance_metrics', {})
                
                ranking_data.append({
                    'strategy': strategy_name,
                    'total_return_pct': summary.get('total_return_pct', 0),
                    'total_trades': summary.get('total_trades', 0),
                    'win_rate': performance.get('win_rate', 0),
                    'profit_factor': performance.get('profit_factor', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'expectancy': performance.get('expectancy', 0)
                })
            
            # Sort by