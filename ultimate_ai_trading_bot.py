#!/usr/bin/env python3
"""
Ultimate AI Trading Bot

Complete system combining:
- Dynamic AI parameter optimization
- Professional confluence analysis
- Advanced risk management
- Real-time notifications
- Comprehensive logging
"""

import asyncio
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import json

# Import our advanced modules
try:
    from dynamic_ai_trading_system import DynamicAITradingBot, AIParameterOptimizer
    from advanced_ai_trading_strategy import AdvancedMarketAnalyzer
    from trade_journal import TradeJournal, TradeRecord
    from notification_system import NotificationManager
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Module import error: {e}")
    MODULES_AVAILABLE = False

class UltimateAITradingBot:
    """Ultimate AI Trading Bot with all advanced features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize all subsystems
        print("🚀 Ultimate AI Trading Bot - Initializing...")
        print("=" * 50)
        
        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")
        
        # Core systems
        self.market_analyzer = AdvancedMarketAnalyzer()
        self.dynamic_trader = DynamicAITradingBot()
        self.ai_optimizer = AIParameterOptimizer()
        
        # Support systems
        self.trade_journal = TradeJournal()
        self.notifications = NotificationManager()
        
        # Trading settings
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        self.scan_interval = self.config.get('scan_interval', 300)  # 5 minutes
        self.max_daily_trades = self.config.get('max_daily_trades', 8)
        
        # Session tracking
        self.session_start = datetime.utcnow()
        self.session_trades = 0
        self.session_pnl = 0.0
        
        print("✅ All systems initialized successfully!")
    
    async def initialize_exchange(self):
        """Initialize exchange for all subsystems"""
        print("🔗 Connecting to exchange...")
        
        success = self.dynamic_trader.initialize_exchange()
        if success:
            self.exchange = self.dynamic_trader.exchange
            print("✅ Exchange connection established")
            return True
        else:
            print("❌ Exchange connection failed")
            return False
    
    async def comprehensive_market_scan(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Perform comprehensive market analysis"""
        try:
            print(f"🔍 Comprehensive analysis: {symbol}")
            
            # Get full market analysis
            market_data = await self.market_analyzer.comprehensive_analysis(
                self.exchange, symbol, '1h'
            )
            
            if not market_data:
                return None
            
            # Generate trading setup
            setup = self.market_analyzer.generate_trading_setup(market_data)
            
            if setup:
                # Add setup data to market_data
                market_data.update({
                    'signal_direction': setup.direction,
                    'signal_strength': setup.confidence,
                    'confluences': setup.confluences,
                    'setup_detected': True
                })
                
                print(f"✅ Trading setup detected: {setup.direction} {symbol}")
                print(f"   Confidence: {setup.confidence:.1f}%")
                print(f"   Confluences: {len(setup.confluences)}")
                
                return market_data
            else:
                print(f"⏳ No setup for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def execute_ai_optimized_trade(self, market_data: Dict[str, Any]) -> bool:
        """Execute trade with full AI optimization"""
        try:
            symbol = market_data['symbol']
            
            print(f"\n🤖 AI-OPTIMIZED TRADE EXECUTION")
            print("=" * 40)
            
            # Get AI parameter optimization
            ai_params = await self.ai_optimizer.get_ai_parameter_analysis(market_data)
            optimized = self.ai_optimizer.optimize_parameters(market_data, ai_params)
            
            # Check AI confidence threshold
            if optimized.confidence_score < 65:
                print(f"⚠️ AI confidence too low: {optimized.confidence_score:.1f}%")
                return False
            
            # Check daily limits
            if self.session_trades >= self.max_daily_trades:
                print(f"🛑 Daily trade limit reached: {self.max_daily_trades}")
                return False
            
            if self.dynamic_trader.consecutive_losses >= 3:
                print(f"🛑 Max consecutive losses reached")
                return False
            
            # Display AI optimization
            self.display_ai_optimization(optimized, market_data)
            
            # Send signal notification
            setup_data = {
                'symbol': symbol,
                'direction': market_data['signal_direction'],
                'entry_price': market_data['current_price'],
                'stop_loss': optimized.stop_loss_level,
                'take_profit': optimized.take_profit_levels[0],  # First TP
                'risk_reward': optimized.risk_reward_ratio,
                'confidence': optimized.confidence_score,
                'timeframe': '1h',
                'confluences': market_data.get('confluences', []),
                'ai_analysis': f"AI-optimized trade with {optimized.leverage}x leverage, {optimized.risk_pct:.1%} risk"
            }
            
            await self.notifications.send_signal_alert(setup_data)
            
            # Execute the trade
            success = await self.dynamic_trader.execute_dynamic_trade(market_data)
            
            if success:
                # Log trade
                trade_record = TradeRecord(
                    trade_id=f"AI_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.utcnow().isoformat(),
                    symbol=symbol,
                    direction=market_data['signal_direction'],
                    entry_price=market_data['current_price'],
                    quantity=optimized.position_size,
                    leverage=optimized.leverage,
                    stop_loss=optimized.stop_loss_level,
                    take_profit=optimized.take_profit_levels[0],
                    risk_reward=optimized.risk_reward_ratio,
                    risk_amount=market_data['current_price'] * optimized.position_size * optimized.risk_pct,
                    confluences=json.dumps(market_data.get('confluences', [])),
                    ai_analysis=market_data.get('ai_insight', ''),
                    signal_strength=optimized.confidence_score,
                    timeframe='1h',
                    rsi_entry=market_data.get('rsi', 50),
                    price_change_24h=market_data.get('price_change_24h', 0),
                    news_sentiment=market_data.get('sentiment_score', 0),
                    status='OPEN'
                )
                
                self.trade_journal.log_trade_entry(trade_record)
                
                # Send execution notification
                trade_data = {
                    'symbol': symbol,
                    'direction': market_data['signal_direction'],
                    'entry_price': market_data['current_price'],
                    'quantity': optimized.position_size,
                    'leverage': optimized.leverage
                }
                
                await self.notifications.send_trade_execution(trade_data)
                
                self.session_trades += 1
                
                print(f"✅ AI-optimized trade executed successfully!")
                return True
            else:
                print(f"❌ Trade execution failed")
                return False
                
        except Exception as e:
            print(f"❌ Error executing AI-optimized trade: {e}")
            return False
    
    def display_ai_optimization(self, optimized, market_data):
        """Display AI optimization details"""
        print(f"\n🧠 AI OPTIMIZATION RESULTS")
        print("-" * 30)
        print(f"🎯 Setup Quality: {optimized.confidence_score:.1f}%")
        print(f"⚡ Trade Style: {optimized.trade_style}")
        print(f"🚨 Urgency: {optimized.urgency}")
        print(f"📈 AI Leverage: {optimized.leverage}x")
        print(f"⚠️ AI Risk: {optimized.risk_pct:.1%}")
        print(f"📊 AI R/R: 1:{optimized.risk_reward_ratio:.1f}")
        
        print(f"\n🎯 AI Reasoning:")
        for reason in optimized.reasoning[:3]:
            print(f"   • {reason}")
    
    async def run_ultimate_trading_session(self, duration_hours: int = 4):
        """Run ultimate AI trading session"""
        print(f"\n🚀 ULTIMATE AI TRADING SESSION")
        print("=" * 50)
        print(f"⏰ Duration: {duration_hours} hours")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
        print(f"🧠 AI-Driven: Leverage 1x-25x, Risk 1%-10%")
        print(f"📊 Max Trades: {self.max_daily_trades}")
        print("")
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        scan_count = 0
        signals_found = 0
        trades_executed = 0
        
        # Send session start notification
        await self.notifications.send_risk_alert(
            "Ultimate AI Session Started",
            f"AI Trading Bot started with dynamic parameters\n"
            f"Duration: {duration_hours}h | Symbols: {len(self.symbols)} | Max trades: {self.max_daily_trades}"
        )
        
        try:
            while datetime.utcnow() < end_time:
                scan_count += 1
                current_time = datetime.utcnow().strftime('%H:%M:%S')
                
                print(f"\n🔍 AI Scan #{scan_count} - {current_time}")
                print("-" * 35)
                
                # Scan all symbols
                for symbol in self.symbols:
                    try:
                        # Comprehensive market analysis
                        market_data = await self.comprehensive_market_scan(symbol)
                        
                        if market_data and market_data.get('setup_detected'):
                            signals_found += 1
                            
                            # Execute AI-optimized trade
                            success = await self.execute_ai_optimized_trade(market_data)
                            
                            if success:
                                trades_executed += 1
                                
                                # Break after successful trade (one at a time)
                                break
                        
                    except Exception as e:
                        print(f"❌ Error processing {symbol}: {e}")
                
                # Show session progress
                elapsed = datetime.utcnow() - self.session_start
                print(f"\n📊 Session Progress:")
                print(f"   ⏱️ Elapsed: {elapsed.total_seconds()/3600:.1f}h")
                print(f"   🔍 Scans: {scan_count}")
                print(f"   🚨 Signals: {signals_found}")
                print(f"   ✅ Trades: {trades_executed}")
                print(f"   💰 Session P&L: ${self.session_pnl:+.2f}")
                
                # Dynamic wait time based on market activity
                if signals_found > trades_executed:
                    wait_time = 180  # 3 minutes if signals pending
                else:
                    wait_time = self.scan_interval  # 5 minutes normal
                
                print(f"⏱️ Next scan in {wait_time//60} minutes...")
                
                # Wait with periodic updates
                for i in range(wait_time):
                    await asyncio.sleep(1)
                    if (i + 1) % 60 == 0:
                        remaining_minutes = (wait_time - i - 1) // 60
                        if remaining_minutes > 0:
                            print(f"   {remaining_minutes} minutes until next scan...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Ultimate AI session stopped by user")
        
        # Final session summary
        total_time = (datetime.utcnow() - self.session_start).total_seconds() / 3600
        
        print(f"\n📋 ULTIMATE AI SESSION SUMMARY")
        print("=" * 40)
        print(f"⏱️ Duration: {total_time:.1f} hours")
        print(f"🔍 Total Scans: {scan_count}")
        print(f"🚨 Signals Found: {signals_found}")
        print(f"✅ Trades Executed: {trades_executed}")
        print(f"💰 Session P&L: ${self.session_pnl:+.2f}")
        print(f"📊 Success Rate: {(trades_executed/signals_found*100):.1f}%" if signals_found > 0 else "No signals")
        
        # Get daily stats
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_stats = self.trade_journal.get_daily_stats(today)
        
        if daily_stats and daily_stats['total_trades'] > 0:
            print(f"\n📈 Daily Performance:")
            print(f"   Trades: {daily_stats['total_trades']}")
            print(f"   Win Rate: {daily_stats['win_rate']:.1f}%")
            print(f"   Total P&L: ${daily_stats['total_pnl']:+.2f}")
            
            # Send daily summary
            await self.notifications.send_daily_summary(daily_stats)
        
        # Send session end notification
        await self.notifications.send_risk_alert(
            "Ultimate AI Session Completed",
            f"Session completed successfully\n"
            f"Trades: {trades_executed} | Signals: {signals_found} | P&L: ${self.session_pnl:+.2f}"
        )

async def main():
    """Main function"""
    print("🤖 Ultimate AI Trading Bot")
    print("=" * 30)
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        print("Please ensure all files are in the same directory")
        return
    
    # Configuration
    config = {
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'scan_interval': 300,  # 5 minutes
        'max_daily_trades': 6
    }
    
    print("\n🎯 Ultimate AI Features:")
    print("• 🧠 AI-optimized leverage (1x-25x)")
    print("• ⚠️ Dynamic risk management (1%-10%)")  
    print("• 📊 Adaptive R/R ratios (1:3-1:5)")
    print("• 🎯 Professional confluences")
    print("• 📱 Real-time notifications")
    print("• 📓 Comprehensive logging")
    
    # Safety confirmation
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    if not testnet:
        print(f"\n🚨 LIVE TRADING MODE DETECTED!")
        print("This system uses HIGH LEVERAGE (up to 25x) and HIGH RISK (up to 10%)")
        print("AI will make autonomous decisions about position sizing and leverage")
        
        confirm = input("\nType 'I UNDERSTAND THE EXTREME RISKS' to proceed: ")
        if confirm != 'I UNDERSTAND THE EXTREME RISKS':
            print("👋 Cancelled for safety")
            return
    
    try:
        # Initialize ultimate bot
        bot = UltimateAITradingBot(config)
        
        if not await bot.initialize_exchange():
            print("❌ Cannot start without exchange connection")
            return
        
        # Get session duration
        duration = int(input("Enter session duration (hours): ") or "2")
        
        # Run ultimate trading session
        await bot.run_ultimate_trading_session(duration)
        
        print("\n✅ Ultimate AI trading session completed!")
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())