import asyncio
import logging
import ccxt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import json
import signal
import sys
from dataclasses import dataclass
from enum import Enum

# Import all components
from strategies.momentum_strategy import MomentumStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.volume_profile_strategy import VolumeProfileStrategy
from strategies.pairs_trading_strategy import PairsTradingStrategy
from strategies.grid_trading_strategy import GridTradingStrategy

from utils.risk_manager import create_risk_manager, RiskLevel
from utils.portfolio_manager import create_portfolio_manager, PositionDirection
from utils.performance_tracker import create_performance_tracker, PerformancePeriod
from utils.market_scanner import create_market_scanner
from utils.notification_manager import create_notification_manager, NotificationType, NotificationPriority
from ai_analysis.claude_analyzer import create_claude_analyzer

from database.connection import get_db_session
from database.models import Trade, SystemLog, LogLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BotState(Enum):
    """Trading bot states"""
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

@dataclass
class BotConfiguration:
    """Bot configuration settings"""
    # Trading settings
    initial_capital: float = 10000.0
    max_open_positions: int = 5
    max_risk_per_trade: float = 0.02  # 2%
    daily_loss_limit: float = 0.05    # 5%
    max_consecutive_losses: int = 3
    
    # Market settings
    symbols: List[str] = None
    primary_timeframe: str = '1h'
    scan_interval: int = 300  # 5 minutes
    
    # AI settings
    use_ai_analysis: bool = True
    ai_confidence_threshold: float = 70.0
    
    # Notification settings
    enable_notifications: bool = True
    notification_channels: List[str] = None
    
    # API settings
    exchange: str = 'binance'
    testnet: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT']
        if self.notification_channels is None:
            self.notification_channels = ['discord', 'email']

class TradingBot:
    """Advanced Cryptocurrency Trading Bot"""
    
    def __init__(self, config: BotConfiguration, api_config, claude_config=None):
        self.config = config
        self.api_config = api_config
        self.claude_config = claude_config
        
        # Bot state
        self.state = BotState.STARTING
        self.start_time = None
        self.last_scan_time = None
        self.shutdown_requested = False
        
        # Core components
        self.exchange = None
        self.strategies = {}
        self.risk_manager = None
        self.portfolio_manager = None
        self.performance_tracker = None
        self.market_scanner = None
        self.notification_manager = None
        self.claude_analyzer = None
        
        # Trading state
        self.active_signals = {}
        self.pending_trades = []
        self.market_data_cache = {}
        
        # Performance metrics
        self.bot_metrics = {
            'total_signals_generated': 0,
            'total_trades_executed': 0,
            'total_ai_analyses': 0,
            'circuit_breaker_activations': 0,
            'last_performance_update': None,
            'uptime_hours': 0
        }
        
        # Event loop
        self.main_loop_task = None
        self.monitoring_task = None
        
        logger.info("Trading Bot initialized")
    
    async def initialize(self):
        """Initialize all bot components"""
        try:
            logger.info("Initializing Trading Bot components...")
            
            # Initialize exchange
            await self._initialize_exchange()
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Initialize risk manager
            await self._initialize_risk_manager()
            
            # Initialize portfolio manager
            await self._initialize_portfolio_manager()
            
            # Initialize performance tracker
            await self._initialize_performance_tracker()
            
            # Initialize market scanner
            await self._initialize_market_scanner()
            
            # Initialize notification manager
            await self._initialize_notification_manager()
            
            # Initialize AI analyzer (optional)
            if self.config.use_ai_analysis and self.claude_config:
                await self._initialize_ai_analyzer()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.state = BotState.RUNNING
            self.start_time = datetime.utcnow()
            
            logger.info("Trading Bot initialization completed successfully")
            
            # Send startup notification
            if self.notification_manager:
                await self.notification_manager.send_custom_notification(
                    title="🚀 Trading Bot Started",
                    message="Trading bot has been initialized and is ready for operation",
                    channels=[],  # Use default channels
                    priority=NotificationPriority.MEDIUM
                )
            
        except Exception as e:
            self.state = BotState.ERROR
            logger.error(f"Failed to initialize Trading Bot: {e}")
            raise
    
    async def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            if self.config.exchange.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': self.api_config.binance_api_key,
                    'secret': self.api_config.binance_secret,
                    'sandbox': self.config.testnet,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.config.exchange}")
            
            await self.exchange.load_markets()
            
            # Test connection
            balance = await self.exchange.fetch_balance()
            logger.info(f"Exchange connected: {self.config.exchange} ({'testnet' if self.config.testnet else 'live'})")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
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
            
            # Grid Trading Strategy (for range-bound markets)
            grid_config = {
                'grid_size': 0.015,  # 1.5% grid spacing
                'num_grids': 8,
                'base_position_size': 0.1,
                'take_profit_pct': 0.012,
                'stop_loss_pct': 0.04
            }
            grid_strategy = GridTradingStrategy(grid_config)
            await grid_strategy.initialize()
            self.strategies['grid_trading'] = grid_strategy
            
            logger.info(f"Initialized {len(self.strategies)} trading strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise
    
    async def _initialize_risk_manager(self):
        """Initialize risk management system"""
        try:
            self.risk_manager = create_risk_manager(self.config)
            logger.info("Risk Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk manager: {e}")
            raise
    
    async def _initialize_portfolio_manager(self):
        """Initialize portfolio management system"""
        try:
            self.portfolio_manager = create_portfolio_manager(self.api_config, self.config)
            await self.portfolio_manager.initialize()
            logger.info("Portfolio Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize portfolio manager: {e}")
            raise
    
    async def _initialize_performance_tracker(self):
        """Initialize performance tracking system"""
        try:
            self.performance_tracker = create_performance_tracker(self.config.initial_capital)
            await self.performance_tracker.initialize()
            logger.info("Performance Tracker initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance tracker: {e}")
            raise
    
    async def _initialize_market_scanner(self):
        """Initialize market scanning system"""
        try:
            self.market_scanner = create_market_scanner(self.api_config)
            await self.market_scanner.initialize()
            logger.info("Market Scanner initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize market scanner: {e}")
            raise
    
    async def _initialize_notification_manager(self):
        """Initialize notification system"""
        try:
            if self.config.enable_notifications:
                self.notification_manager = create_notification_manager(self.api_config)
                await self.notification_manager.initialize()
                logger.info("Notification Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification manager: {e}")
            # Don't raise - notifications are not critical
    
    async def _initialize_ai_analyzer(self):
        """Initialize AI analysis system"""
        try:
            if self.claude_config:
                self.claude_analyzer = create_claude_analyzer(self.claude_config.api_key)
                await self.claude_analyzer.initialize()
                logger.info("Claude AI Analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI analyzer: {e}")
            # Don't raise - AI analysis is optional
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Main bot execution loop"""
        try:
            logger.info("Starting Trading Bot main execution loop...")
            
            # Start main trading loop
            self.main_loop_task = asyncio.create_task(self._main_trading_loop())
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Wait for tasks to complete or shutdown signal
            await asyncio.gather(
                self.main_loop_task,
                self.monitoring_task,
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error in main bot execution: {e}")
            self.state = BotState.ERROR
        finally:
            await self._cleanup()
    
    async def _main_trading_loop(self):
        """Main trading logic loop"""
        logger.info("Main trading loop started")
        
        while not self.shutdown_requested and self.state == BotState.RUNNING:
            try:
                loop_start_time = datetime.utcnow()
                
                # Check circuit breakers
                circuit_status = await self.risk_manager.check_circuit_breakers()
                if circuit_status['triggered']:
                    logger.warning(f"Circuit breaker triggered: {circuit_status['reason']}")
                    
                    # Send notification
                    if self.notification_manager:
                        await self.notification_manager.send_circuit_breaker_alert({
                            'trigger_type': circuit_status['triggers'][0]['type'] if circuit_status['triggers'] else 'UNKNOWN',
                            'current_value': circuit_status['triggers'][0]['value'] if circuit_status['triggers'] else 'Unknown',
                            'threshold': circuit_status['triggers'][0]['threshold'] if circuit_status['triggers'] else 'Unknown',
                            'severity': circuit_status.get('severity', 'HIGH'),
                            'cooldown_until': circuit_status.get('cooldown_until', 'Unknown')
                        })
                    
                    self.bot_metrics['circuit_breaker_activations'] += 1
                    
                    # Pause trading during circuit breaker
                    await asyncio.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Update portfolio snapshot
                await self._update_portfolio_snapshot()
                
                # Scan markets and generate signals
                await self._scan_and_generate_signals()
                
                # Process pending trades
                await self._process_pending_trades()
                
                # Check stop loss and take profit
                await self._check_position_exits()
                
                # Update bot metrics
                self._update_bot_metrics()
                
                # Calculate loop duration and sleep
                loop_duration = (datetime.utcnow() - loop_start_time).total_seconds()
                sleep_time = max(0, self.config.scan_interval - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                self.last_scan_time = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
        
        logger.info("Main trading loop stopped")
    
    async def _monitoring_loop(self):
        """System monitoring and maintenance loop"""
        logger.info("Monitoring loop started")
        
        while not self.shutdown_requested and self.state == BotState.RUNNING:
            try:
                # Update performance metrics every hour
                if (datetime.utcnow().minute == 0 and 
                    (not self.bot_metrics['last_performance_update'] or
                     datetime.utcnow() - self.bot_metrics['last_performance_update'] > timedelta(hours=1))):
                    
                    await self._update_performance_metrics()
                    await self._send_performance_update()
                
                # Daily summary at midnight
                if datetime.utcnow().hour == 0 and datetime.utcnow().minute < 5:
                    await self._send_daily_summary()
                
                # Health checks
                await self._perform_health_checks()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
        
        logger.info("Monitoring loop stopped")
    
    async def _scan_and_generate_signals(self):
        """Scan markets and generate trading signals"""
        try:
            # Get market opportunities
            opportunities = await self.market_scanner.scan_for_opportunities(
                custom_criteria={'max_symbols_to_scan': len(self.config.symbols)}
            )
            
            for opportunity in opportunities:
                symbol = opportunity.symbol
                
                if symbol not in self.config.symbols:
                    continue
                
                # Get market data for the symbol
                market_data = await self._get_market_data(symbol)
                if market_data is None or len(market_data) < 50:
                    continue
                
                current_price = market_data['close'].iloc[-1]
                
                # Generate signals from all strategies
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = await strategy.generate_signal(symbol, market_data, current_price)
                        
                        if signal and signal.signal_type in ['BUY', 'SELL', 'LONG', 'SHORT']:
                            self.bot_metrics['total_signals_generated'] += 1
                            
                            # Evaluate signal with risk manager
                            portfolio_context = await self._get_portfolio_context()
                            
                            risk_assessment = await self.risk_manager.evaluate_trade_proposal(
                                symbol, signal.__dict__, portfolio_context
                            )
                            
                            # Filter by risk level
                            if risk_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]:
                                
                                # AI analysis (if enabled)
                                if self.claude_analyzer and signal.confidence >= self.config.ai_confidence_threshold:
                                    ai_analysis = await self._get_ai_analysis(symbol, signal, market_data)
                                    
                                    if ai_analysis and ai_analysis.confidence >= self.config.ai_confidence_threshold:
                                        signal.confidence = (signal.confidence + ai_analysis.confidence) / 2
                                        signal.reasoning.extend(ai_analysis.key_insights)
                                        self.bot_metrics['total_ai_analyses'] += 1
                                
                                # Add to pending trades
                                trade_proposal = {
                                    'symbol': symbol,
                                    'strategy': strategy_name,
                                    'signal': signal,
                                    'risk_assessment': risk_assessment,
                                    'ai_analysis': ai_analysis if self.claude_analyzer else None,
                                    'timestamp': datetime.utcnow(),
                                    'market_data': market_data.tail(1).to_dict('records')[0]
                                }
                                
                                self.pending_trades.append(trade_proposal)
                                
                                logger.info(f"Signal generated: {strategy_name} - {symbol} {signal.signal_type} "
                                          f"(Confidence: {signal.confidence:.1f}%, Risk: {risk_assessment.risk_level.value})")
                    
                    except Exception as e:
                        logger.warning(f"Error generating signal for {symbol} with {strategy_name}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error in market scanning: {e}")
    
    async def _process_pending_trades(self):
        """Process pending trade proposals"""
        try:
            if not self.pending_trades:
                return
            
            # Sort by confidence and risk score
            self.pending_trades.sort(
                key=lambda x: (x['signal'].confidence, -x['risk_assessment'].risk_score),
                reverse=True
            )
            
            # Process top trades within position limits
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            available_positions = self.config.max_open_positions - portfolio_summary.total_positions
            
            trades_to_execute = []
            
            for i, trade_proposal in enumerate(self.pending_trades):
                if len(trades_to_execute) >= available_positions:
                    break
                
                # Check if we already have a position in this symbol
                if trade_proposal['symbol'] in [pos for pos in await self._get_open_positions()]:
                    continue
                
                trades_to_execute.append(trade_proposal)
            
            # Execute trades
            for trade_proposal in trades_to_execute:
                success = await self._execute_trade(trade_proposal)
                if success:
                    self.bot_metrics['total_trades_executed'] += 1
            
            # Clear processed trades
            self.pending_trades = [
                trade for trade in self.pending_trades 
                if trade not in trades_to_execute
            ]
            
            # Remove old pending trades (older than 1 hour)
            current_time = datetime.utcnow()
            self.pending_trades = [
                trade for trade in self.pending_trades
                if (current_time - trade['timestamp']).total_seconds() < 3600
            ]
            
        except Exception as e:
            logger.error(f"Error processing pending trades: {e}")
    
    async def _execute_trade(self, trade_proposal: Dict[str, Any]) -> bool:
        """Execute a trade proposal"""
        try:
            symbol = trade_proposal['symbol']
            signal = trade_proposal['signal']
            risk_assessment = trade_proposal['risk_assessment']
            
            # Calculate position size
            position_size = risk_assessment.recommended_position_size
            
            if position_size <= 0:
                logger.warning(f"Invalid position size for {symbol}: {position_size}")
                return False
            
            # Determine direction
            direction = PositionDirection.LONG if signal.signal_type in ['BUY', 'LONG'] else PositionDirection.SHORT
            
            # Execute trade through portfolio manager
            success = await self.portfolio_manager.add_position(
                symbol=symbol,
                direction=direction,
                quantity=position_size,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if success:
                # Log trade
                logger.info(f"Trade executed: {symbol} {direction.value} {position_size} @ ${signal.entry_price:.4f}")
                
                # Send notification
                if self.notification_manager:
                    await self.notification_manager.send_trade_notification({
                        'symbol': symbol,
                        'direction': direction.value,
                        'quantity': position_size,
                        'entry_price': signal.entry_price,
                        'strategy_id': trade_proposal['strategy'],
                        'confidence': signal.confidence,
                        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                    })
                
                return True
            else:
                logger.warning(f"Failed to execute trade for {symbol}")
                return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def _check_position_exits(self):
        """Check for stop loss and take profit triggers"""
        try:
            triggers = await self.portfolio_manager.check_stop_loss_take_profit()
            
            for trigger in triggers:
                symbol = trigger['symbol']
                trigger_type = trigger['type']
                current_price = trigger['current_price']
                
                # Close position
                success = await self.portfolio_manager.close_position(
                    symbol=symbol,
                    exit_price=current_price
                )
                
                if success:
                    logger.info(f"{trigger_type} triggered for {symbol} at ${current_price:.4f}")
                    
                    # Add trade to performance tracker
                    trade_data = {
                        'symbol': symbol,
                        'direction': trigger['direction'],
                        'quantity': trigger['quantity'],
                        'entry_price': 0,  # Will be filled by portfolio manager
                        'exit_price': current_price,
                        'pnl': trigger['unrealized_pnl'],
                        'entry_time': datetime.utcnow() - timedelta(hours=1),  # Estimate
                        'exit_time': datetime.utcnow(),
                        'exit_reason': trigger_type
                    }
                    
                    await self.performance_tracker.add_completed_trade(trade_data)
                    
                    # Send notification
                    if self.notification_manager:
                        if trigger_type == 'STOP_LOSS':
                            await self.notification_manager.send_trade_notification({
                                'symbol': symbol,
                                'direction': trigger['direction'],
                                'quantity': trigger['quantity'],
                                'exit_price': current_price,
                                'pnl': trigger['unrealized_pnl'],
                                'exit_reason': 'STOP_LOSS'
                            })
                        elif trigger_type == 'TAKE_PROFIT':
                            await self.notification_manager.send_trade_notification({
                                'symbol': symbol,
                                'direction': trigger['direction'],
                                'quantity': trigger['quantity'],
                                'exit_price': current_price,
                                'pnl': trigger['unrealized_pnl'],
                                'exit_reason': 'TAKE_PROFIT'
                            })
            
        except Exception as e:
            logger.error(f"Error checking position exits: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for a symbol with caching"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{self.config.primary_timeframe}"
            current_time = datetime.utcnow()
            
            if cache_key in self.market_data_cache:
                cached_data, cache_time = self.market_data_cache[cache_key]
                if (current_time - cache_time).total_seconds() < 300:  # 5 minute cache
                    return cached_data
            
            # Fetch new data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                self.config.primary_timeframe, 
                limit=100
            )
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Cache the data
                self.market_data_cache[cache_key] = (df, current_time)
                
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def _get_portfolio_context(self) -> Dict[str, Any]:
        """Get current portfolio context"""
        try:
            summary = await self.portfolio_manager.get_portfolio_summary()
            
            return {
                'total_value': summary.total_value,
                'available_balance': summary.available_balance,
                'current_positions': await self._get_open_positions(),
                'daily_pnl': summary.daily_pnl,
                'risk_tolerance': 'MEDIUM'  # Could be dynamic based on performance
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio context: {e}")
            return {}
    
    async def _get_open_positions(self) -> List[str]:
        """Get list of symbols with open positions"""
        try:
            return list(self.portfolio_manager.positions.keys())
        except:
            return []
    
    async def _get_ai_analysis(self, symbol: str, signal, market_data: pd.DataFrame):
        """Get AI analysis for signal"""
        try:
            if not self.claude_analyzer:
                return None
            
            current_price = market_data['close'].iloc[-1]
            
            ai_analysis = await self.claude_analyzer.analyze_trading_opportunity(
                symbol=symbol,
                market_data=market_data,
                current_price=current_price,
                strategy_signals=[{
                    'strategy_id': 'combined',
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning
                }]
            )
            
            return ai_analysis
            
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return None
    
    async def _update_portfolio_snapshot(self):
        """Update portfolio snapshot for performance tracking"""
        try:
            summary = await self.portfolio_manager.get_portfolio_summary()
            
            await self.performance_tracker.update_portfolio_snapshot(
                portfolio_value=summary.total_value,
                cash_balance=summary.cash_balance,
                invested_amount=summary.invested_amount,
                active_positions=summary.total_positions,
                daily_pnl=summary.daily_pnl
            )
            
        except Exception as e:
            logger.error(f"Error updating portfolio snapshot: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Save daily snapshot
            await self.portfolio_manager.save_daily_snapshot()
            
            # Update timestamp
            self.bot_metrics['last_performance_update'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _send_performance_update(self):
        """Send performance update notification"""
        try:
            if not self.notification_manager:
                return
            
            summary = await self.performance_tracker.get_performance_summary(PerformancePeriod.DAILY)
            
            if summary.get('total_trades', 0) > 0:
                await self.notification_manager.send_custom_notification(
                    title="📊 Hourly Performance Update",
                    message=f"Portfolio: ${summary['current_portfolio_value']:,.2f}\n"
                           f"Daily P&L: ${summary['daily_pnl']:+,.2f}\n"
                           f"Total Return: {summary['total_return']:+.2f}%\n"
                           f"Active Positions: {summary['total_positions']}",
                    channels=[],  # Use default
                    priority=NotificationPriority.LOW
                )
            
        except Exception as e:
            logger.error(f"Error sending performance update: {e}")
    
    async def _send_daily_summary(self):
        """Send daily performance summary"""
        try:
            if not self.notification_manager:
                return
            
            # Get performance data
            summary = await self.performance_tracker.get_performance_summary(PerformancePeriod.DAILY)
            weekly_summary = await self.performance_tracker.get_performance_summary(PerformancePeriod.WEEKLY)
            monthly_summary = await self.performance_tracker.get_performance_summary(PerformancePeriod.MONTHLY)
            
            # Send daily summary
            await self.notification_manager.send_daily_summary(
                portfolio_summary={
                    'total_value': summary['current_portfolio_value'],
                    'total_pnl': summary['total_pnl'],
                    'total_positions': summary['total_positions']
                },
                performance_metrics={
                    'daily_return': summary.get('daily_return', 0),
                    'weekly_return': weekly_summary.get('total_return', 0),
                    'monthly_return': monthly_summary.get('total_return', 0),
                    'sharpe_ratio': summary.get('sharpe_ratio', 0),
                    'max_drawdown': summary.get('max_drawdown', 0)
                },
                trade_summary={
                    'trades_today': self.bot_metrics['total_trades_executed'],
                    'win_rate': summary.get('win_rate', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    async def _perform_health_checks(self):
        """Perform system health checks"""
        try:
            # Check component health
            health_issues = []
            
            # Portfolio manager health
            portfolio_health = await self.portfolio_manager.get_portfolio_health_check()
            if portfolio_health['status'] != 'healthy':
                health_issues.extend(portfolio_health.get('warnings', []))
            
            # Risk manager health
            risk_health = await self.risk_manager.get_risk_health_check()
            if risk_health['status'] != 'healthy':
                health_issues.extend(risk_health.get('warnings', []))
            
            # Performance tracker health
            if self.performance_tracker:
                perf_stats = self.performance_tracker.get_statistics()
                if perf_stats.get('total_snapshots', 0) == 0:
                    health_issues.append("No performance snapshots recorded")
            
            # Market scanner health
            if self.market_scanner:
                scanner_health = await self.market_scanner.health_check()
                if not scanner_health.get('exchange_connected', False):
                    health_issues.append("Market scanner exchange connection issue")
            
            # Notification manager health
            if self.notification_manager:
                notif_health = await self.notification_manager.get_health_status()
                if notif_health['status'] != 'healthy':
                    health_issues.extend(notif_health.get('warnings', []))
            
            # Report critical health issues
            if health_issues:
                logger.warning(f"Health issues detected: {health_issues}")
                
                if self.notification_manager:
                    await self.notification_manager.send_custom_notification(
                        title="⚠️ System Health Warning",
                        message=f"Health issues detected:\n" + "\n".join(f"• {issue}" for issue in health_issues[:5]),
                        channels=[],
                        priority=NotificationPriority.HIGH
                    )
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    def _update_bot_metrics(self):
        """Update bot performance metrics"""
        try:
            if self.start_time:
                self.bot_metrics['uptime_hours'] = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            
        except Exception as e:
            logger.error(f"Error updating bot metrics: {e}")
    
    async def get_bot_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        try:
            # Portfolio summary
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            
            # Performance summary
            performance_summary = await self.performance_tracker.get_performance_summary()
            
            # Risk summary
            risk_summary = await self.risk_manager.get_risk_summary()
            
            status = {
                'bot_state': self.state.value,
                'uptime_hours': self.bot_metrics['uptime_hours'],
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
                
                # Configuration
                'configuration': {
                    'symbols': self.config.symbols,
                    'max_positions': self.config.max_open_positions,
                    'scan_interval': self.config.scan_interval,
                    'use_ai_analysis': self.config.use_ai_analysis,
                    'testnet': self.config.testnet
                },
                
                # Portfolio status
                'portfolio': {
                    'total_value': portfolio_summary.total_value,
                    'cash_balance': portfolio_summary.cash_balance,
                    'daily_pnl': portfolio_summary.daily_pnl,
                    'total_return': portfolio_summary.total_pnl_pct,
                    'active_positions': portfolio_summary.total_positions,
                    'largest_position_pct': portfolio_summary.largest_position_pct
                },
                
                # Performance metrics
                'performance': {
                    'total_return': performance_summary.get('total_return', 0),
                    'sharpe_ratio': performance_summary.get('sharpe_ratio', 0),
                    'max_drawdown': performance_summary.get('max_drawdown', 0),
                    'win_rate': performance_summary.get('win_rate', 0),
                    'total_trades': performance_summary.get('total_trades', 0)
                },
                
                # Risk metrics
                'risk': {
                    'current_drawdown': risk_summary.get('risk_metrics', {}).get('current_drawdown', 0),
                    'circuit_breaker_status': risk_summary.get('circuit_breaker_status', {}).get('status', 'UNKNOWN'),
                    'consecutive_losses': risk_summary.get('consecutive_losses', 0)
                },
                
                # Bot metrics
                'bot_metrics': self.bot_metrics.copy(),
                
                # Component status
                'components': {
                    'strategies': len(self.strategies),
                    'risk_manager': self.risk_manager is not None,
                    'portfolio_manager': self.portfolio_manager is not None,
                    'performance_tracker': self.performance_tracker is not None,
                    'market_scanner': self.market_scanner is not None,
                    'notification_manager': self.notification_manager is not None,
                    'ai_analyzer': self.claude_analyzer is not None
                },
                
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {'error': str(e)}
    
    async def pause_trading(self):
        """Pause trading operations"""
        try:
            if self.state == BotState.RUNNING:
                self.state = BotState.PAUSED
                logger.info("Trading operations paused")
                
                if self.notification_manager:
                    await self.notification_manager.send_custom_notification(
                        title="⏸️ Trading Paused",
                        message="Trading operations have been paused",
                        channels=[],
                        priority=NotificationPriority.MEDIUM
                    )
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error pausing trading: {e}")
            return False
    
    async def resume_trading(self):
        """Resume trading operations"""
        try:
            if self.state == BotState.PAUSED:
                self.state = BotState.RUNNING
                logger.info("Trading operations resumed")
                
                if self.notification_manager:
                    await self.notification_manager.send_custom_notification(
                        title="▶️ Trading Resumed",
                        message="Trading operations have been resumed",
                        channels=[],
                        priority=NotificationPriority.MEDIUM
                    )
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error resuming trading: {e}")
            return False
    
    async def emergency_stop(self):
        """Emergency stop - close all positions"""
        try:
            logger.warning("Emergency stop initiated - closing all positions")
            
            # Close all open positions
            closed_positions = []
            for symbol in await self._get_open_positions():
                success = await self.portfolio_manager.close_position(symbol)
                if success:
                    closed_positions.append(symbol)
            
            # Pause trading
            await self.pause_trading()
            
            # Send notification
            if self.notification_manager:
                await self.notification_manager.send_custom_notification(
                    title="🚨 EMERGENCY STOP",
                    message=f"Emergency stop executed. Closed {len(closed_positions)} positions: {', '.join(closed_positions)}",
                    channels=[],
                    priority=NotificationPriority.CRITICAL
                )
            
            logger.warning(f"Emergency stop completed. Closed {len(closed_positions)} positions")
            return True
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
            return False
    
    async def force_close_position(self, symbol: str):
        """Force close a specific position"""
        try:
            success = await self.portfolio_manager.close_position(symbol)
            
            if success:
                logger.info(f"Force closed position: {symbol}")
                
                if self.notification_manager:
                    await self.notification_manager.send_custom_notification(
                        title="🔧 Position Force Closed",
                        message=f"Position {symbol} has been force closed",
                        channels=[],
                        priority=NotificationPriority.MEDIUM
                    )
                
                return True
            else:
                logger.warning(f"Failed to force close position: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error force closing position {symbol}: {e}")
            return False
    
    async def update_strategy_config(self, strategy_name: str, config: Dict[str, Any]):
        """Update strategy configuration"""
        try:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                
                # Update configuration
                for key, value in config.items():
                    if hasattr(strategy, key):
                        setattr(strategy, key, value)
                
                logger.info(f"Updated configuration for strategy: {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy not found: {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating strategy config: {e}")
            return False
    
    async def _cleanup(self):
        """Cleanup resources and shut down gracefully"""
        try:
            logger.info("Starting bot cleanup...")
            self.state = BotState.STOPPING
            
            # Cancel running tasks
            if self.main_loop_task and not self.main_loop_task.done():
                self.main_loop_task.cancel()
                try:
                    await self.main_loop_task
                except asyncio.CancelledError:
                    pass
            
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup components
            cleanup_tasks = []
            
            if self.portfolio_manager:
                cleanup_tasks.append(self.portfolio_manager.cleanup())
            
            if self.performance_tracker:
                cleanup_tasks.append(self.performance_tracker.cleanup())
            
            if self.market_scanner:
                cleanup_tasks.append(self.market_scanner.cleanup())
            
            if self.notification_manager:
                cleanup_tasks.append(self.notification_manager.cleanup())
            
            if self.claude_analyzer:
                cleanup_tasks.append(self.claude_analyzer.cleanup())
            
            if self.exchange:
                cleanup_tasks.append(self.exchange.close())
            
            # Wait for all cleanup tasks
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Send shutdown notification
            if self.notification_manager and self.state != BotState.ERROR:
                try:
                    uptime = self.bot_metrics['uptime_hours']
                    await self.notification_manager.send_custom_notification(
                        title="🛑 Trading Bot Shutdown",
                        message=f"Trading bot has shut down gracefully.\n"
                               f"Uptime: {uptime:.1f} hours\n"
                               f"Total trades: {self.bot_metrics['total_trades_executed']}\n"
                               f"Total signals: {self.bot_metrics['total_signals_generated']}",
                        channels=[],
                        priority=NotificationPriority.MEDIUM
                    )
                except:
                    pass  # Don't fail cleanup on notification error
            
            self.state = BotState.STOPPED
            logger.info("Bot cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.state = BotState.ERROR

# Factory function
def create_trading_bot(config: BotConfiguration, api_config, claude_config=None) -> TradingBot:
    """Factory function to create Trading Bot instance"""
    return TradingBot(config, api_config, claude_config)

# Example configuration classes for testing
class ExampleApiConfig:
    def __init__(self):
        # Exchange API
        self.binance_api_key = "your_binance_api_key"
        self.binance_secret = "your_binance_secret"
        self.binance_testnet = True
        
        # Notification APIs
        self.discord_webhook_url = "your_discord_webhook"
        self.telegram_bot_token = "your_telegram_bot_token"
        self.telegram_chat_id = "your_telegram_chat_id"
        self.email_user = "your_email@gmail.com"
        self.email_password = "your_app_password"
        self.to_emails = ["recipient@gmail.com"]

class ExampleClaudeConfig:
    def __init__(self):
        self.api_key = "your_claude_api_key"

# Example usage
async def main():
    """Example main function"""
    try:
        # Configuration
        bot_config = BotConfiguration(
            initial_capital=10000.0,
            max_open_positions=5,
            symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
            testnet=True,
            use_ai_analysis=True
        )
        
        api_config = ExampleApiConfig()
        claude_config = ExampleClaudeConfig()
        
        # Create and initialize bot
        bot = create_trading_bot(bot_config, api_config, claude_config)
        await bot.initialize()
        
        print("🤖 Trading Bot initialized successfully!")
        print(f"📊 Strategies: {list(bot.strategies.keys())}")
        print(f"💱 Symbols: {bot.config.symbols}")
        print(f"💰 Initial capital: ${bot.config.initial_capital:,.2f}")
        print(f"🔧 Configuration: Testnet={bot.config.testnet}, AI={bot.config.use_ai_analysis}")
        
        # Get initial status
        status = await bot.get_bot_status()
        print(f"\n📈 Initial Status:")
        print(f"  Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
        print(f"  Cash Balance: ${status['portfolio']['cash_balance']:,.2f}")
        print(f"  Active Positions: {status['portfolio']['active_positions']}")
        
        # Run bot
        print(f"\n🚀 Starting trading operations...")
        await bot.run()
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")
    finally:
        try:
            if 'bot' in locals():
                await bot._cleanup()
        except:
            pass

if __name__ == "__main__":
    # Run the trading bot
    asyncio.run(main())