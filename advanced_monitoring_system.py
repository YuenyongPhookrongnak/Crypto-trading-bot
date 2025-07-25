#!/usr/bin/env python3
"""
Advanced Monitoring and Alert System

ระบบ monitoring ที่ครอบคลุม:
- Drawdown alerts และ auto-close
- Performance monitoring
- System health checks
- Real-time notifications
- Emergency stop mechanisms
"""

import asyncio
import ccxt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class DrawdownLevel(Enum):
    MILD = 5      # 5% drawdown
    MODERATE = 10 # 10% drawdown
    SEVERE = 15   # 15% drawdown
    CRITICAL = 20 # 20% drawdown
    EMERGENCY = 25 # 25% drawdown

@dataclass
class AlertConfig:
    """Alert configuration"""
    # Drawdown alerts
    mild_drawdown_pct: float = 5.0
    moderate_drawdown_pct: float = 10.0
    severe_drawdown_pct: float = 15.0
    critical_drawdown_pct: float = 20.0
    emergency_drawdown_pct: float = 25.0
    
    # Auto-close settings
    auto_close_at_severe: bool = True
    auto_close_at_critical: bool = True
    emergency_shutdown: bool = True
    
    # Performance alerts
    consecutive_losses_limit: int = 5
    daily_loss_limit_pct: float = 8.0
    win_rate_warning_threshold: float = 30.0
    
    # System health
    api_error_threshold: int = 10
    connection_timeout: float = 30.0
    memory_usage_threshold: float = 80.0
    
    # Notification settings
    discord_webhook: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""
    email_alerts: bool = False

@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0

@dataclass
class SystemHealth:
    """System health metrics"""
    uptime: timedelta = timedelta()
    api_errors_count: int = 0
    last_api_error: Optional[datetime] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    connection_status: str = "Connected"
    last_heartbeat: datetime = datetime.utcnow()
    active_positions: int = 0
    pending_orders: int = 0

class DrawdownMonitor:
    """Advanced drawdown monitoring and management"""
    
    def __init__(self, config: AlertConfig, initial_balance: float):
        self.config = config
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        
        self.drawdown_history = []
        self.alert_sent = {level: False for level in DrawdownLevel}
        
    def update_balance(self, new_balance: float) -> Optional[Dict[str, Any]]:
        """Update balance and check for drawdown alerts"""
        self.current_balance = new_balance
        
        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            # Reset alert flags on new peak
            self.alert_sent = {level: False for level in DrawdownLevel}
        
        # Calculate current drawdown
        current_drawdown_pct = ((self.peak_balance - new_balance) / self.peak_balance) * 100
        
        # Record drawdown
        self.drawdown_history.append({
            'timestamp': datetime.utcnow(),
            'balance': new_balance,
            'peak_balance': self.peak_balance,
            'drawdown_pct': current_drawdown_pct
        })
        
        # Keep only last 1000 records
        if len(self.drawdown_history) > 1000:
            self.drawdown_history = self.drawdown_history[-1000:]
        
        # Check for alerts
        return self._check_drawdown_alerts(current_drawdown_pct)
    
    def _check_drawdown_alerts(self, current_drawdown: float) -> Optional[Dict[str, Any]]:
        """Check and generate drawdown alerts"""
        alert_data = None
        
        # Emergency level (25%)
        if current_drawdown >= self.config.emergency_drawdown_pct and not self.alert_sent[DrawdownLevel.EMERGENCY]:
            alert_data = {
                'level': AlertLevel.EMERGENCY,
                'drawdown_level': DrawdownLevel.EMERGENCY,
                'current_drawdown': current_drawdown,
                'action': 'EMERGENCY_SHUTDOWN' if self.config.emergency_shutdown else 'ALERT_ONLY',
                'message': f'🚨 EMERGENCY: {current_drawdown:.1f}% drawdown reached!'
            }
            self.alert_sent[DrawdownLevel.EMERGENCY] = True
            
        # Critical level (20%)
        elif current_drawdown >= self.config.critical_drawdown_pct and not self.alert_sent[DrawdownLevel.CRITICAL]:
            alert_data = {
                'level': AlertLevel.CRITICAL,
                'drawdown_level': DrawdownLevel.CRITICAL,
                'current_drawdown': current_drawdown,
                'action': 'AUTO_CLOSE_ALL' if self.config.auto_close_at_critical else 'ALERT_ONLY',
                'message': f'🔴 CRITICAL: {current_drawdown:.1f}% drawdown - immediate action required!'
            }
            self.alert_sent[DrawdownLevel.CRITICAL] = True
            
        # Severe level (15%)
        elif current_drawdown >= self.config.severe_drawdown_pct and not self.alert_sent[DrawdownLevel.SEVERE]:
            alert_data = {
                'level': AlertLevel.CRITICAL,
                'drawdown_level': DrawdownLevel.SEVERE,
                'current_drawdown': current_drawdown,
                'action': 'AUTO_CLOSE_ALL' if self.config.auto_close_at_severe else 'ALERT_ONLY',
                'message': f'🟠 SEVERE: {current_drawdown:.1f}% drawdown detected!'
            }
            self.alert_sent[DrawdownLevel.SEVERE] = True
            
        # Moderate level (10%)
        elif current_drawdown >= self.config.moderate_drawdown_pct and not self.alert_sent[DrawdownLevel.MODERATE]:
            alert_data = {
                'level': AlertLevel.WARNING,
                'drawdown_level': DrawdownLevel.MODERATE,
                'current_drawdown': current_drawdown,
                'action': 'REDUCE_RISK',
                'message': f'🟡 MODERATE: {current_drawdown:.1f}% drawdown - consider reducing risk'
            }
            self.alert_sent[DrawdownLevel.MODERATE] = True
            
        # Mild level (5%)
        elif current_drawdown >= self.config.mild_drawdown_pct and not self.alert_sent[DrawdownLevel.MILD]:
            alert_data = {
                'level': AlertLevel.INFO,
                'drawdown_level': DrawdownLevel.MILD,
                'current_drawdown': current_drawdown,
                'action': 'MONITOR',
                'message': f'🔵 MILD: {current_drawdown:.1f}% drawdown - monitoring closely'
            }
            self.alert_sent[DrawdownLevel.MILD] = True
        
        return alert_data
    
    def get_drawdown_stats(self) -> Dict[str, Any]:
        """Get comprehensive drawdown statistics"""
        if not self.drawdown_history:
            return {}
        
        drawdowns = [record['drawdown_pct'] for record in self.drawdown_history]
        
        return {
            'current_drawdown': drawdowns[-1] if drawdowns else 0,
            'max_drawdown': max(drawdowns) if drawdowns else 0,
            'avg_drawdown': sum(drawdowns) / len(drawdowns) if drawdowns else 0,
            'peak_balance': self.peak_balance,
            'current_balance': self.current_balance,
            'balance_from_peak': ((self.current_balance - self.peak_balance) / self.peak_balance) * 100,
            'recovery_needed_pct': ((self.peak_balance - self.current_balance) / self.current_balance) * 100
        }

class PerformanceAnalyzer:
    """Advanced performance analysis and monitoring"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.trade_history = []
        self.daily_stats = {}
        
    def add_trade_result(self, pnl: float, is_win: bool) -> Optional[Dict[str, Any]]:
        """Add trade result and check for performance alerts"""
        # Update basic metrics
        self.metrics.total_trades += 1
        self.metrics.total_pnl += pnl
        
        if is_win:
            self.metrics.winning_trades += 1
            self.metrics.consecutive_wins += 1
            self.metrics.consecutive_losses = 0
        else:
            self.metrics.losing_trades += 1
            self.metrics.consecutive_losses += 1
            self.metrics.consecutive_wins = 0
        
        # Calculate win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = (self.metrics.winning_trades / self.metrics.total_trades) * 100
        
        # Add to trade history
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'pnl': pnl,
            'is_win': is_win,
            'consecutive_losses': self.metrics.consecutive_losses
        })
        
        # Check for performance alerts
        return self._check_performance_alerts()
    
    def _check_performance_alerts(self) -> Optional[Dict[str, Any]]:
        """Check for performance-based alerts"""
        # Consecutive losses alert
        if self.metrics.consecutive_losses >= self.config.consecutive_losses_limit:
            return {
                'level': AlertLevel.WARNING,
                'type': 'CONSECUTIVE_LOSSES',
                'value': self.metrics.consecutive_losses,
                'message': f'⚠️ {self.metrics.consecutive_losses} consecutive losses detected!'
            }
        
        # Win rate alert (only after minimum trades)
        if (self.metrics.total_trades >= 20 and 
            self.metrics.win_rate < self.config.win_rate_warning_threshold):
            return {
                'level': AlertLevel.WARNING,
                'type': 'LOW_WIN_RATE',
                'value': self.metrics.win_rate,
                'message': f'📉 Low win rate: {self.metrics.win_rate:.1f}% (threshold: {self.config.win_rate_warning_threshold}%)'
            }
        
        # Daily loss limit
        today = datetime.utcnow().strftime('%Y-%m-%d')
        today_pnl = sum(trade['pnl'] for trade in self.trade_history 
                       if trade['timestamp'].strftime('%Y-%m-%d') == today)
        
        if today_pnl < 0:
            daily_loss_pct = abs(today_pnl / self.metrics.current_balance) * 100
            if daily_loss_pct >= self.config.daily_loss_limit_pct:
                return {
                    'level': AlertLevel.CRITICAL,
                    'type': 'DAILY_LOSS_LIMIT',
                    'value': daily_loss_pct,
                    'message': f'🔴 Daily loss limit exceeded: {daily_loss_pct:.1f}%'
                }
        
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if self.metrics.total_trades == 0:
            return {'error': 'No trades to analyze'}
        
        # Calculate profit factor
        winning_trades_pnl = sum(trade['pnl'] for trade in self.trade_history if trade['is_win'])
        losing_trades_pnl = abs(sum(trade['pnl'] for trade in self.trade_history if not trade['is_win']))
        
        profit_factor = winning_trades_pnl / losing_trades_pnl if losing_trades_pnl > 0 else float('inf')
        
        # Calculate average win/loss
        avg_win = winning_trades_pnl / self.metrics.winning_trades if self.metrics.winning_trades > 0 else 0
        avg_loss = losing_trades_pnl / self.metrics.losing_trades if self.metrics.losing_trades > 0 else 0
        
        # Recent performance (last 10 trades)
        recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
        recent_wins = sum(1 for trade in recent_trades if trade['is_win'])
        recent_win_rate = (recent_wins / len(recent_trades)) * 100 if recent_trades else 0
        
        return {
            'total_trades': self.metrics.total_trades,
            'win_rate': self.metrics.win_rate,
            'total_pnl': self.metrics.total_pnl,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'consecutive_losses': self.metrics.consecutive_losses,
            'consecutive_wins': self.metrics.consecutive_wins,
            'recent_win_rate': recent_win_rate,
            'recent_trades_count': len(recent_trades),
            'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else 0
        }

class SystemHealthMonitor:
    """System health and connectivity monitoring"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.health = SystemHealth()
        self.start_time = datetime.utcnow()
        self.api_errors = []
        
    def record_api_error(self, error_type: str, error_message: str) -> Optional[Dict[str, Any]]:
        """Record API error and check thresholds"""
        self.health.api_errors_count += 1
        self.health.last_api_error = datetime.utcnow()
        
        # Add to error history
        self.api_errors.append({
            'timestamp': datetime.utcnow(),
            'type': error_type,
            'message': error_message
        })
        
        # Keep only recent errors (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.api_errors = [error for error in self.api_errors if error['timestamp'] > cutoff_time]
        
        # Check error threshold
        recent_errors = len(self.api_errors)
        if recent_errors >= self.config.api_error_threshold:
            return {
                'level': AlertLevel.WARNING,
                'type': 'API_ERRORS',
                'value': recent_errors,
                'message': f'⚠️ High API error rate: {recent_errors} errors in 24h'
            }
        
        return None
    
    def update_system_stats(self, memory_usage: float, cpu_usage: float, 
                           active_positions: int, pending_orders: int) -> Optional[Dict[str, Any]]:
        """Update system statistics"""
        self.health.memory_usage = memory_usage
        self.health.cpu_usage = cpu_usage
        self.health.active_positions = active_positions
        self.health.pending_orders = pending_orders
        self.health.last_heartbeat = datetime.utcnow()
        self.health.uptime = datetime.utcnow() - self.start_time
        
        # Check memory usage
        if memory_usage >= self.config.memory_usage_threshold:
            return {
                'level': AlertLevel.WARNING,
                'type': 'HIGH_MEMORY_USAGE',
                'value': memory_usage,
                'message': f'⚠️ High memory usage: {memory_usage:.1f}%'
            }
        
        return None
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        recent_errors = len([e for e in self.api_errors 
                           if e['timestamp'] > datetime.utcnow() - timedelta(hours=1)])
        
        return {
            'uptime_hours': self.health.uptime.total_seconds() / 3600,
            'memory_usage': self.health.memory_usage,
            'cpu_usage': self.health.cpu_usage,
            'api_errors_24h': len(self.api_errors),
            'api_errors_1h': recent_errors,
            'last_error': self.health.last_api_error.isoformat() if self.health.last_api_error else None,
            'connection_status': self.health.connection_status,
            'active_positions': self.health.active_positions,
            'pending_orders': self.health.pending_orders,
            'last_heartbeat': self.health.last_heartbeat.isoformat()
        }

class NotificationManager:
    """Advanced notification management"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.notification_history = []
        
    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert through configured channels"""
        message = self._format_alert_message(alert_data)
        
        # Send to Discord
        if self.config.discord_webhook:
            await self._send_discord_alert(message, alert_data['level'])
        
        # Send to Telegram
        if self.config.telegram_token and self.config.telegram_chat_id:
            await self._send_telegram_alert(message)
        
        # Record notification
        self.notification_history.append({
            'timestamp': datetime.utcnow(),
            'level': alert_data['level'],
            'message': message,
            'alert_data': alert_data
        })
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert message"""
        level = alert_data['level'].value
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🤖 **Trading Bot Alert** - {level}\n"
        message += f"⏰ Time: {timestamp}\n"
        message += f"📄 Message: {alert_data['message']}\n"
        
        if 'current_drawdown' in alert_data:
            message += f"📉 Drawdown: {alert_data['current_drawdown']:.1f}%\n"
        
        if 'action' in alert_data:
            message += f"🎯 Action: {alert_data['action']}\n"
        
        return message
    
    async def _send_discord_alert(self, message: str, level: AlertLevel):
        """Send alert to Discord"""
        try:
            import aiohttp
            
            color_map = {
                AlertLevel.INFO: 0x3498db,      # Blue
                AlertLevel.WARNING: 0xf39c12,   # Orange
                AlertLevel.CRITICAL: 0xe74c3c,  # Red
                AlertLevel.EMERGENCY: 0x8b0000  # Dark Red
            }
            
            embed = {
                "title": f"Trading Bot Alert - {level.value}",
                "description": message,
                "color": color_map.get(level, 0x3498db),
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "AI Trading Bot Monitoring System"}
            }
            
            payload = {"embeds": [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.discord_webhook, json=payload) as response:
                    if response.status == 204:
                        print("✅ Discord alert sent")
                    else:
                        print(f"❌ Discord alert failed: {response.status}")
                        
        except Exception as e:
            print(f"❌ Discord notification error: {e}")
    
    async def _send_telegram_alert(self, message: str):
        """Send alert to Telegram"""
        try:
            import aiohttp
            
            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        print("✅ Telegram alert sent")
                    else:
                        print(f"❌ Telegram alert failed: {response.status}")
                        
        except Exception as e:
            print(f"❌ Telegram notification error: {e}")

class TradingBotMonitor:
    """Complete trading bot monitoring system"""
    
    def __init__(self, initial_balance: float, config: AlertConfig = None):
        self.config = config or AlertConfig()
        
        # Initialize monitoring components
        self.drawdown_monitor = DrawdownMonitor(self.config, initial_balance)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        self.health_monitor = SystemHealthMonitor(self.config)
        self.notification_manager = NotificationManager(self.config)
        
        # Emergency flags
        self.emergency_stop_triggered = False
        self.auto_trading_disabled = False
        
        print("🔍 Advanced Trading Bot Monitor initialized")
        print(f"   Initial balance: ${initial_balance:,.2f}")
        print(f"   Drawdown thresholds: {self.config.mild_drawdown_pct}% / {self.config.severe_drawdown_pct}% / {self.config.emergency_drawdown_pct}%")
    
    async def update_balance(self, new_balance: float) -> bool:
        """Update balance and check for alerts"""
        # Update drawdown monitoring
        drawdown_alert = self.drawdown_monitor.update_balance(new_balance)
        
        if drawdown_alert:
            await self.notification_manager.send_alert(drawdown_alert)
            
            # Handle emergency actions
            if drawdown_alert.get('action') == 'EMERGENCY_SHUTDOWN':
                self.emergency_stop_triggered = True
                return False  # Signal to stop trading
            elif drawdown_alert.get('action') == 'AUTO_CLOSE_ALL':
                self.auto_trading_disabled = True
                return False  # Signal to close positions
        
        return True  # Continue trading
    
    async def record_trade(self, pnl: float, is_win: bool):
        """Record trade result and check performance"""
        perf_alert = self.performance_analyzer.add_trade_result(pnl, is_win)
        
        if perf_alert:
            await self.notification_manager.send_alert(perf_alert)
    
    async def record_system_event(self, event_type: str, **kwargs):
        """Record system events and check health"""
        if event_type == 'api_error':
            health_alert = self.health_monitor.record_api_error(
                kwargs.get('error_type', 'Unknown'),
                kwargs.get('error_message', '')
            )
            if health_alert:
                await self.notification_manager.send_alert(health_alert)
        
        elif event_type == 'system_stats':
            health_alert = self.health_monitor.update_system_stats(
                kwargs.get('memory_usage', 0),
                kwargs.get('cpu_usage', 0),
                kwargs.get('active_positions', 0),
                kwargs.get('pending_orders', 0)
            )
            if health_alert:
                await self.notification_manager.send_alert(health_alert)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'emergency_stop': self.emergency_stop_triggered,
            'auto_trading_disabled': self.auto_trading_disabled,
            'drawdown_stats': self.drawdown_monitor.get_drawdown_stats(),
            'performance_report': self.performance_analyzer.get_performance_report(),
            'system_health': self.health_monitor.get_health_report(),
            'alert_config': {
                'drawdown_thresholds': {
                    'mild': self.config.mild_drawdown_pct,
                    'moderate': self.config.moderate_drawdown_pct,
                    'severe': self.config.severe_drawdown_pct,
                    'critical': self.config.critical_drawdown_pct,
                    'emergency': self.config.emergency_drawdown_pct
                },
                'auto_close_enabled': self.config.auto_close_at_severe,
                'emergency_shutdown': self.config.emergency_shutdown
            }
        }
    
    async def send_daily_summary(self):
        """Send daily performance summary"""
        status = self.get_comprehensive_status()
        
        summary_alert = {
            'level': AlertLevel.INFO,
            'message': f"📊 Daily Summary\n"
                      f"Balance: ${self.drawdown_monitor.current_balance:,.2f}\n"
                      f"Drawdown: {status['drawdown_stats'].get('current_drawdown', 0):.1f}%\n"
                      f"Trades: {status['performance_report'].get('total_trades', 0)}\n"
                      f"Win Rate: {status['performance_report'].get('win_rate', 0):.1f}%\n"
                      f"Total P&L: ${status['performance_report'].get('total_pnl', 0):.2f}"
        }
        
        await self.notification_manager.send_alert(summary_alert)

# Demo function
async def demo_monitoring_system():
    """Demo monitoring and alert system"""
    print("🔍 Advanced Monitoring System Demo")
    print("=" * 40)
    
    # Setup configuration
    config = AlertConfig(
        mild_drawdown_pct=3.0,
        moderate_drawdown_pct=7.0,
        severe_drawdown_pct=12.0,
        critical_drawdown_pct=18.0,
        emergency_drawdown_pct=25.0,
        auto_close_at_severe=True,
        consecutive_losses_limit=4,
        daily_loss_limit_pct=6.0
    )
    
    # Initialize monitor
    monitor = TradingBotMonitor(10000.0, config)
    
    print("📊 Simulating trading scenarios:")
    
    # Scenario 1: Normal trading
    print("\n1. Normal Trading Scenario")
    await monitor.update_balance(10150.0)  # +1.5%
    await monitor.record_trade(150.0, True)
    await monitor.update_balance(10080.0)  # Small loss
    await monitor.record_trade(-70.0, False)
    
    # Scenario 2: Moderate drawdown
    print("\n2. Moderate Drawdown Scenario")
    await monitor.update_balance(9300.0)  # -7% drawdown
    await monitor.record_trade(-780.0, False)
    
    # Scenario 3: Consecutive losses
    print("\n3. Consecutive Losses Scenario")
    for i in range(5):
        await monitor.record_trade(-50.0, False)
    
    # Scenario 4: System health issues
    print("\n4. System Health Issues")
    await monitor.record_system_event('api_error', 
                                     error_type='ConnectionError', 
                                     error_message='Exchange connection timeout')
    
    await monitor.record_system_event('system_stats',
                                     memory_usage=85.0,
                                     cpu_usage=45.0,
                                     active_positions=2,
                                     pending_orders=3)
    
    # Get comprehensive status
    print("\n📋 Final Status Report:")
    status = monitor.get_comprehensive_status()
    
    print(f"Emergency Stop: {status['emergency_stop']}")
    print(f"Auto Trading: {'Disabled' if status['auto_trading_disabled'] else 'Enabled'}")
    print(f"Current Drawdown: {status['drawdown_stats'].get('current_drawdown', 0):.1f}%")
    print(f"Total Trades: {status['performance_report'].get('total_trades', 0)}")
    print(f"Win Rate: {status['performance_report'].get('win_rate', 0):.1f}%")
    print(f"Consecutive Losses: {status['performance_report'].get('consecutive_losses', 0)}")
    
    print("\n✅ Monitoring system demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_monitoring_system())