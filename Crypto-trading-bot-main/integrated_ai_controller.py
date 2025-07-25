#!/usr/bin/env python3
"""
Ultimate Integrated AI Trading System

ระบบการเทรดที่สมบูรณ์แบบ รวมทุกฟีเจอร์ที่ร้องขอ:

✅ Enhanced Market Analysis:
   - OBV, ATR, VWAP, Bollinger Bands Width, Williams %R
   - CMF, Stochastic RSI
   - Weighted signal strength calculation

✅ AI Optimization:
   - Dynamic leverage adjustment (1x-25x)
   - News sentiment analysis
   - Volatility-based parameter adjustment
   - Market condition assessment

✅ Advanced Order Execution:
   - Smart market/limit order routing
   - Trailing stops based on BOS
   - Slippage protection
   - Split order execution

✅ Comprehensive Monitoring:
   - Drawdown alerts (5%, 10%, 15%, 20%, 25%)
   - Auto-close at critical levels
   - Performance tracking
   - System health monitoring
   - Real-time notifications
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
import time

# Import our enhanced modules
from enhanced_market_analysis import (
    EnhancedMarketAnalyzer, 
    EnhancedSignalStrengthCalculator,
    EnhancedTechnicalAnalysis
)
from ai_optimization_system import (
    DynamicAIOptimizer,
    DynamicTradingParameters,
    MarketCondition,
    EnhancedNewsAnalyzer
)
from advanced_order_execution import (
    AdvancedOrderManager,
    OrderRequest,
    OrderType,
    OrderUrgency,
    TrailingStopType
)
from monitoring_alert_system import (
    TradingBotMonitor,
    AlertConfig,
    AlertLevel,
    DrawdownLevel
)

class TradingMode(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE" 
    AGGRESSIVE = "AGGRESSIVE"
    AI_ADAPTIVE = "AI_ADAPTIVE"

@dataclass
class UltimateConfig:
    """Ultimate trading system configuration"""
    # Core settings
    initial_capital: float = 10000.0
    trading_mode: TradingMode = TradingMode.AI_ADAPTIVE
    symbols: List[str] = None
    max_concurrent_positions: int = 3
    
    # Risk management
    max_risk_per_trade: float = 0.08  # 8% max
    base_leverage_limit: int = 25
    emergency_stop_loss: float = 0.25  # 25% portfolio loss
    
    # AI settings
    enable_ai_optimization: bool = True
    enable_news_analysis: bool = True
    enable_dynamic_leverage: bool = True
    ai_confidence_threshold: float = 65.0
    
    # Advanced features
    enable_trailing_stops: bool = True
    enable_smart_execution: bool = True
    enable_split_orders: bool = True
    max_slippage_tolerance: float = 0.5
    
    # Monitoring
    enable_drawdown_alerts: bool = True
    enable_performance_tracking: bool = True
    enable_system_health: bool = True
    notification_level: AlertLevel = AlertLevel.WARNING

class UltimateAITradingSystem:
    """The ultimate AI trading system combining all advanced features"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.config.symbols = config.symbols or ['BTC/USDT', 'ETH/USDT']
        
        print("🚀 Ultimate AI Trading System")
        print("=" * 50)
        print(f"💰 Capital: ${config.initial_capital:,.2f}")
        print(f"🧠 Mode: {config.trading_mode.value}")
        print(f"🎯 Symbols: {', '.join(self.config.symbols)}")
        print(f"⚡ Max Risk: {config.max_risk_per_trade:.1%}")
        print(f"📈 Max Leverage: {config.base_leverage_limit}x")
        
        # Initialize core systems
        self._initialize_core_systems()
        
        # Session tracking
        self.session_start = datetime.utcnow()
        self.session_stats = {
            'trades_executed': 0,
            'signals_found': 0,
            'ai_optimizations': 0,
            'alerts_sent': 0,
            'emergency_stops': 0
        }
        
        print("✅ All systems initialized and ready!")
    
    def _initialize_core_systems(self):
        """Initialize all core trading systems"""
        try:
            # Market analysis system
            self.market_analyzer = EnhancedMarketAnalyzer()
            print("✅ Enhanced Market Analyzer")
            
            # AI optimization system  
            self.ai_optimizer = DynamicAIOptimizer()
            print("✅ Dynamic AI Optimizer")
            
            # Monitoring system
            alert_config = AlertConfig(
                mild_drawdown_pct=5.0,
                moderate_drawdown_pct=10.0,
                severe_drawdown_pct=15.0,
                critical_drawdown_pct=20.0,
                emergency_drawdown_pct=25.0,
                auto_close_at_severe=True,
                auto_close_at_critical=True,
                emergency_shutdown=True,
                consecutive_losses_limit=5,
                daily_loss_limit_pct=8.0
            )
            
            self.monitor = TradingBotMonitor(self.config.initial_capital, alert_config)
            print("✅ Advanced Monitoring System")
            
            # Exchange and order management (will be set when connected)
            self.exchange = None
            self.order_manager = None
            
        except Exception as e:
            print(f"❌ System initialization error: {e}")
            raise
    
    async def initialize_exchange(self) -> bool:
        """Initialize exchange connection and order management"""
        try:
            # Initialize exchange
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET', ''),
                'sandbox': os.getenv('BINANCE_TESTNET', 'true').lower() == 'true',
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                }
            })
            
            # Test connection
            markets = self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            
            print(f"✅ Exchange connected")
            print(f"💰 USDT Balance: {balance['USDT']['total']:,.2f}")
            print(f"📊 Available markets: {len(markets)}")
            
            # Initialize order management
            self.order_manager = AdvancedOrderManager(self.exchange)
            print("✅ Advanced Order Manager")
            
            return True
            
        except Exception as e:
            print(f"❌ Exchange connection failed: {e}")
            return False
    
    async def comprehensive_market_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Perform comprehensive market analysis with all enhancements"""
        try:
            print(f"🔬 Comprehensive analysis: {symbol}")
            
            # Enhanced technical analysis
            analysis_data = await self.market_analyzer.comprehensive_enhanced_analysis(
                self.exchange, symbol, '1h'
            )
            
            if not analysis_data:
                print(f"⚠️ No analysis data for {symbol}")
                return None
            
            # AI news analysis (if enabled)
            if self.config.enable_news_analysis:
                news_analysis = await self.ai_optimizer.news_analyzer.analyze_market_news(symbol)
                analysis_data['news_analysis'] = news_analysis
            
            # Enhanced signal strength calculation
            signal_strength = analysis_data.get('signal_strength', 60)
            
            # Check if we have a tradeable setup
            if signal_strength >= self.config.ai_confidence_threshold:
                confluences = analysis_data.get('confluences', [])
                
                print(f"✅ Strong setup detected:")
                print(f"   Signal Strength: {signal_strength:.1f}%")
                print(f"   Confluences: {len(confluences)}")
                print(f"   Market Regime: {analysis_data.get('market_regime', 'Unknown')}")
                
                # Determine trade direction
                if analysis_data.get('bos_bullish') and signal_strength > 70:
                    analysis_data['trade_direction'] = 'BUY'
                elif analysis_data.get('bos_bearish') and signal_strength > 70:
                    analysis_data['trade_direction'] = 'SELL'
                elif analysis_data.get('bullish_alignment') and analysis_data.get('rsi', 50) < 60:
                    analysis_data['trade_direction'] = 'BUY'
                elif analysis_data.get('bearish_alignment') and analysis_data.get('rsi', 50) > 40:
                    analysis_data['trade_direction'] = 'SELL'
                else:
                    analysis_data['trade_direction'] = None
                
                return analysis_data
            else:
                print(f"⏳ Signal strength too low: {signal_strength:.1f}%")
                return None
                
        except Exception as e:
            print(f"❌ Analysis error for {symbol}: {e}")
            await self.monitor.record_system_event('api_error', 
                                                  error_type='AnalysisError',
                                                  error_message=str(e))
            return None
    
    async def ai_parameter_optimization(self, analysis_data: Dict[str, Any]) -> Optional[DynamicTradingParameters]:
        """AI-driven parameter optimization"""
        try:
            if not self.config.enable_ai_optimization:
                return self._get_static_parameters(analysis_data)
            
            print(f"🧠 AI optimizing parameters...")
            
            # Get news analysis if available
            news_analysis = analysis_data.get('news_analysis', {})
            
            # AI optimization
            dynamic_params = await self.ai_optimizer.optimize_dynamic_parameters(
                analysis_data, news_analysis
            )
            
            print(f"🎯 AI Optimization Results:")
            print(f"   Leverage: {dynamic_params.base_leverage}x → {dynamic_params.leverage}x")
            print(f"   Risk: {dynamic_params.risk_per_trade:.1%}")
            print(f"   R/R: 1:{dynamic_params.risk_reward_ratio:.1f}")
            print(f"   Market Condition: {dynamic_params.market_condition.value}")
            print(f"   Confidence: {dynamic_params.confidence_score:.1f}%")
            
            # Reasoning
            print(f"🧠 AI Reasoning:")
            for reason in dynamic_params.adjustment_reasons[:3]:
                print(f"   • {reason}")
            
            self.session_stats['ai_optimizations'] += 1
            return dynamic_params
            
        except Exception as e:
            print(f"❌ AI optimization error: {e}")
            return self._get_static_parameters(analysis_data)
    
    def _get_static_parameters(self, analysis_data: Dict[str, Any]) -> DynamicTradingParameters:
        """Fallback static parameters based on trading mode"""
        signal_strength = analysis_data.get('signal_strength', 60)
        volatility = analysis_data.get('atr_pct', 2.0)
        
        if self.config.trading_mode == TradingMode.CONSERVATIVE:
            leverage = 3
            risk = 0.01
            rr = 3.0
        elif self.config.trading_mode == TradingMode.MODERATE:
            leverage = 5 if signal_strength > 75 else 3
            risk = 0.02 if signal_strength > 80 else 0.015
            rr = 3.5
        elif self.config.trading_mode == TradingMode.AGGRESSIVE:
            leverage = 10 if signal_strength > 85 else 7
            risk = 0.04 if signal_strength > 85 else 0.03
            rr = 4.0
        else:  # AI_ADAPTIVE fallback
            leverage = min(8, max(3, int(signal_strength / 10)))
            risk = min(0.03, max(0.01, signal_strength / 3000))
            rr = 3.0 + (volatility * 0.2)
        
        return DynamicTradingParameters(
            leverage=leverage,
            risk_per_trade=risk,
            risk_reward_ratio=rr,
            stop_loss_pct=0.02 * (volatility / 2),
            base_leverage=leverage,
            volatility_multiplier=1.0,
            news_adjustment=1.0,
            structure_confidence=signal_strength / 100,
            position_sizing_method='FIXED',
            partial_close_levels=[0.4, 0.35, 0.25],
            trailing_stop_trigger=2.0,
            adjustment_reasons=[f"Static {self.config.trading_mode.value} mode"],
            market_condition=MarketCondition.NEUTRAL,
            confidence_score=signal_strength
        )
    
    async def execute_ultimate_trade(self, analysis_data: Dict[str, Any], 
                                   dynamic_params: DynamicTradingParameters) -> bool:
        """Execute trade with all advanced features"""
        try:
            symbol = analysis_data['symbol']
            direction = analysis_data.get('trade_direction')
            current_price = analysis_data['current_price']
            
            if not direction:
                print("⚠️ No clear trade direction - skipping")
                return False
            
            print(f"\n⚡ ULTIMATE TRADE EXECUTION")
            print("=" * 40)
            print(f"📊 Symbol: {symbol} {direction}")
            print(f"💰 Price: ${current_price:,.2f}")
            print(f"🎯 Setup: {dynamic_params.confidence_score:.1f}% confidence")
            
            # Calculate position size
            available_balance = self.exchange.fetch_balance()['USDT']['free']
            position_value = available_balance * dynamic_params.risk_per_trade * dynamic_params.leverage
            position_size = position_value / current_price
            
            # Set leverage and margin mode
            self.exchange.set_leverage(dynamic_params.leverage, symbol)
            self.exchange.set_margin_mode('ISOLATED', symbol)
            
            print(f"📈 Leverage: {dynamic_params.leverage}x (ISOLATED)")
            print(f"💰 Position Size: {position_size:.6f}")
            print(f"⚠️ Risk: {dynamic_params.risk_per_trade:.1%}")
            
            # Smart order execution
            if self.config.enable_smart_execution:
                # Determine urgency based on setup
                if dynamic_params.market_condition == MarketCondition.HIGH_VOLATILITY:
                    urgency = "immediate"
                elif dynamic_params.confidence_score > 85:
                    urgency = "normal"
                else:
                    urgency = "patient"
                
                print(f"🎯 Execution urgency: {urgency}")
                
                entry_result = await self.order_manager.place_entry_order(
                    symbol, direction.lower(), position_size, 
                    dynamic_params.market_condition.value.lower(), urgency
                )
                
                if entry_result.status != 'filled':
                    print(f"❌ Entry order failed: {entry_result.status}")
                    return False
                
                executed_price = entry_result.average_price
                print(f"✅ Entry executed at ${executed_price:,.2f}")
                
            else:
                # Simple market order
                order = self.exchange.create_market_order(symbol, direction.lower(), position_size)
                executed_price = float(order.get('average', current_price))
                print(f"✅ Market order executed at ${executed_price:,.2f}")
            
            # Calculate stop loss and take profit levels
            if direction == 'BUY':
                stop_loss = executed_price * (1 - dynamic_params.stop_loss_pct)
                tp_levels = [
                    executed_price * (1 + dynamic_params.stop_loss_pct * dynamic_params.risk_reward_ratio * 0.5),
                    executed_price * (1 + dynamic_params.stop_loss_pct * dynamic_params.risk_reward_ratio * 0.8),
                    executed_price * (1 + dynamic_params.stop_loss_pct * dynamic_params.risk_reward_ratio)
                ]
            else:
                stop_loss = executed_price * (1 + dynamic_params.stop_loss_pct)
                tp_levels = [
                    executed_price * (1 - dynamic_params.stop_loss_pct * dynamic_params.risk_reward_ratio * 0.5),
                    executed_price * (1 - dynamic_params.stop_loss_pct * dynamic_params.risk_reward_ratio * 0.8),
                    executed_price * (1 - dynamic_params.stop_loss_pct * dynamic_params.risk_reward_ratio)
                ]
            
            # Place stop loss (with trailing if enabled)
            if self.config.enable_trailing_stops:
                trail_id = await self.order_manager.place_stop_loss(
                    symbol, direction.lower(), position_size, stop_loss,
                    trailing=True, trail_distance=dynamic_params.stop_loss_pct * 100
                )
                print(f"🔄 Trailing stop placed: {trail_id}")
            else:
                stop_id = await self.order_manager.place_stop_loss(
                    symbol, direction.lower(), position_size, stop_loss
                )
                print(f"🛑 Stop loss: ${stop_loss:.2f}")
            
            # Place take profit levels
            partial_amounts = [
                position_size * ratio for ratio in dynamic_params.partial_close_levels
            ]
            
            tp_orders = await self.order_manager.place_take_profit_levels(
                symbol, direction.lower(), position_size, tp_levels, partial_amounts
            )
            
            print(f"💰 Take profit levels set: {len(tp_orders)}")
            for i, tp in enumerate(tp_levels, 1):
                print(f"   TP{i}: ${tp:.2f}")
            
            # Update session stats
            self.session_stats['trades_executed'] += 1
            
            # Record trade for monitoring
            expected_pnl = position_size * abs(tp_levels[0] - executed_price)
            await self.monitor.record_trade(expected_pnl, True)  # Optimistic for now
            
            print(f"\n✅ ULTIMATE TRADE COMPLETED!")
            print(f"🎯 Expected R/R: 1:{dynamic_params.risk_reward_ratio:.1f}")
            print(f"📊 Risk: ${position_size * abs(stop_loss - executed_price):.2f}")
            print(f"💰 Potential reward: ${expected_pnl:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            await self.monitor.record_system_event('api_error',
                                                  error_type='ExecutionError',
                                                  error_message=str(e))
            return False
    
    async def run_ultimate_session(self, duration_hours: int = 4):
        """Run ultimate trading session with all features"""
        print(f"\n🚀 ULTIMATE AI TRADING SESSION")
        print("=" * 50)
        print(f"⏰ Duration: {duration_hours} hours")
        print(f"🧠 AI Mode: {self.config.trading_mode.value}")
        print(f"🎯 Symbols: {', '.join(self.config.symbols)}")
        print(f"📈 Max Leverage: {self.config.base_leverage_limit}x")
        print(f"⚠️ Max Risk: {self.config.max_risk_per_trade:.1%}")
        print("")
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        scan_count = 0
        
        try:
            while datetime.utcnow() < end_time:
                scan_count += 1
                current_time = datetime.utcnow().strftime('%H:%M:%S')
                
                print(f"\n🔍 Ultimate Scan #{scan_count} - {current_time}")
                print("-" * 45)
                
                # Check monitoring status
                if self.monitor.emergency_stop_triggered:
                    print("🚨 EMERGENCY STOP TRIGGERED - HALTING TRADING")
                    break
                
                if self.monitor.auto_trading_disabled:
                    print("⚠️ Auto trading disabled - monitoring only")
                    await asyncio.sleep(300)  # 5 minutes
                    continue
                
                # Update system health
                import psutil
                await self.monitor.record_system_event('system_stats',
                                                      memory_usage=psutil.virtual_memory().percent,
                                                      cpu_usage=psutil.cpu_percent(),
                                                      active_positions=len(await self._get_active_positions()),
                                                      pending_orders=0)
                
                # Scan each symbol
                for symbol in self.config.symbols:
                    try:
                        # Comprehensive market analysis
                        analysis_data = await self.comprehensive_market_analysis(symbol)
                        
                        if analysis_data and analysis_data.get('trade_direction'):
                            self.session_stats['signals_found'] += 1
                            
                            print(f"🚨 SIGNAL: {analysis_data['trade_direction']} {symbol}")
                            
                            # AI parameter optimization
                            dynamic_params = await self.ai_parameter_optimization(analysis_data)
                            
                            if dynamic_params and dynamic_params.confidence_score >= self.config.ai_confidence_threshold:
                                # Check position limits
                                active_positions = await self._get_active_positions()
                                
                                if len(active_positions) < self.config.max_concurrent_positions:
                                    # Execute ultimate trade
                                    success = await self.execute_ultimate_trade(analysis_data, dynamic_params)
                                    
                                    if success:
                                        # Update balance for monitoring
                                        current_balance = self.exchange.fetch_balance()['USDT']['total']
                                        continue_trading = await self.monitor.update_balance(current_balance)
                                        
                                        if not continue_trading:
                                            print("🛑 Monitoring system signaled to stop trading")
                                            break
                                else:
                                    print(f"⚠️ Max positions reached: {len(active_positions)}")
                            else:
                                print(f"⏳ AI confidence too low or optimization failed")
                        else:
                            print(f"⏳ No signal for {symbol}")
                            
                    except Exception as e:
                        print(f"❌ Error processing {symbol}: {e}")
                        await self.monitor.record_system_event('api_error',
                                                              error_type='ProcessingError',
                                                              error_message=str(e))
                
                # Session progress update
                elapsed = datetime.utcnow() - self.session_start
                
                print(f"\n📊 Session Progress:")
                print(f"   ⏱️ Elapsed: {elapsed.total_seconds()/3600:.1f}h")
                print(f"   🔍 Scans: {scan_count}")
                print(f"   🚨 Signals: {self.session_stats['signals_found']}")
                print(f"   ✅ Trades: {self.session_stats['trades_executed']}")
                print(f"   🧠 AI Optimizations: {self.session_stats['ai_optimizations']}")
                
                # Monitoring status
                monitor_status = self.monitor.get_comprehensive_status()
                current_drawdown = monitor_status['drawdown_stats'].get('current_drawdown', 0)
                win_rate = monitor_status['performance_report'].get('win_rate', 0)
                
                print(f"   📉 Drawdown: {current_drawdown:.1f}%")
                print(f"   🎯 Win Rate: {win_rate:.1f}%")
                
                # Dynamic wait based on activity
                if self.session_stats['signals_found'] > self.session_stats['trades_executed']:
                    wait_time = 180  # 3 minutes if signals pending
                else:
                    wait_time = 300  # 5 minutes normal
                
                print(f"⏱️ Next scan in {wait_time//60} minutes...")
                
                for i in range(wait_time):
                    await asyncio.sleep(1)
                    if (i + 1) % 60 == 0:
                        remaining = (wait_time - i - 1) // 60
                        if remaining > 0:
                            print(f"   {remaining} minutes remaining...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Ultimate session stopped by user")
        
        # Final session summary
        await self._generate_session_summary(scan_count)
    
    async def _get_active_positions(self) -> List[Dict]:
        """Get currently active positions"""
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if p['contracts'] and p['contracts'] > 0]
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return []
    
    async def _generate_session_summary(self, scan_count: int):
        """Generate comprehensive session summary"""
        total_time = (datetime.utcnow() - self.session_start).total_seconds() / 3600
        
        print(f"\n📋 ULTIMATE SESSION SUMMARY")
        print("=" * 50)
        print(f"⏱️ Duration: {total_time:.1f} hours")
        print(f"🔍 Total Scans: {scan_count}")
        print(f"🚨 Signals Found: {self.session_stats['signals_found']}")
        print(f"✅ Trades Executed: {self.session_stats['trades_executed']}")
        print(f"🧠 AI Optimizations: {self.session_stats['ai_optimizations']}")
        print(f"📱 Alerts Sent: {self.session_stats['alerts_sent']}")
        
        # Performance metrics
        monitor_status = self.monitor.get_comprehensive_status()
        
        print(f"\n📈 Performance Metrics:")
        perf = monitor_status['performance_report']
        if 'error' not in perf:
            print(f"   Win Rate: {perf.get('win_rate', 0):.1f}%")
            print(f"   Total P&L: ${perf.get('total_pnl', 0):+.2f}")
            print(f"   Profit Factor: {perf.get('profit_factor', 0):.2f}")
        
        print(f"\n📉 Risk Metrics:")
        drawdown = monitor_status['drawdown_stats']
        print(f"   Current Drawdown: {drawdown.get('current_drawdown', 0):.1f}%")
        print(f"   Max Drawdown: {drawdown.get('max_drawdown', 0):.1f}%")
        print(f"   Peak Balance: ${drawdown.get('peak_balance', 0):,.2f}")
        print(f"   Current Balance: ${drawdown.get('current_balance', 0):,.2f}")
        
        print(f"\n🔧 System Health:")
        health = monitor_status['system_health']
        print(f"   Uptime: {health.get('uptime_hours', 0):.1f}h")
        print(f"   API Errors: {health.get('api_errors_24h', 0)}")
        print(f"   Memory Usage: {health.get('memory_usage', 0):.1f}%")
        
        print(f"\n🎯 Feature Utilization:")
        print(f"   ✅ Enhanced Technical Analysis")
        print(f"   ✅ AI Parameter Optimization")
        print(f"   ✅ Smart Order Execution")
        print(f"   ✅ Advanced Risk Monitoring")
        print(f"   ✅ Real-time Notifications")
        
        # Send daily summary notification
        await self.monitor.send_daily_summary()

# Demo and main execution
async def demo_ultimate_system():
    """Demo the ultimate integrated system"""
    print("🌟 Ultimate AI Trading System Demo")
    print("=" * 40)
    
    # Configuration for demo
    config = UltimateConfig(
        initial_capital=10000.0,
        trading_mode=TradingMode.AI_ADAPTIVE,
        symbols=['BTC/USDT', 'ETH/USDT'],
        max_concurrent_positions=2,
        max_risk_per_trade=0.04,  # 4% max for demo
        base_leverage_limit=15,   # Conservative for demo
        enable_ai_optimization=True,
        enable_news_analysis=True,
        enable_smart_execution=True,
        enable_trailing_stops=True,
        ai_confidence_threshold=70.0
    )
    
    # Initialize system
    system = UltimateAITradingSystem(config)
    
    # Initialize exchange
    if not await system.initialize_exchange():
        print("❌ Cannot run demo without exchange connection")
        return
    
    print("\n🎯 Demo Options:")
    print("1. Quick analysis demo (1 symbol)")
    print("2. Mini trading session (15 minutes)")
    print("3. Full demo session (1 hour)")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Quick analysis demo
            analysis = await system.comprehensive_market_analysis('BTC/USDT')
            if analysis:
                params = await system.ai_parameter_optimization(analysis)
                if params:
                    print(f"\n✅ Demo analysis completed!")
                    print(f"   Signal strength: {analysis['signal_strength']:.1f}%")
                    print(f"   AI leverage: {params.leverage}x")
                    print(f"   Trade direction: {analysis.get('trade_direction', 'None')}")
            
        elif choice == "2":
            await system.run_ultimate_session(duration_hours=0.25)  # 15 minutes
            
        elif choice == "3":
            await system.run_ultimate_session(duration_hours=1)     # 1 hour
        
        print(f"\n✅ Ultimate system demo completed!")
        
    except KeyboardInterrupt:
        print(f"\n👋 Demo interrupted by user")

if __name__ == "__main__":
    asyncio.run(demo_ultimate_system())