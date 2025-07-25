#!/usr/bin/env python3
"""
Advanced Trade Journal System

Comprehensive logging and analysis system for AI trading bot
with Notion-style organization and performance analytics.
"""

import pandas as pd
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import sqlite3

@dataclass
class TradeRecord:
    """Complete trade record structure"""
    trade_id: str
    timestamp: str
    symbol: str
    direction: str  # BUY/SELL
    entry_price: float
    exit_price: float = 0.0
    quantity: float = 0.0
    leverage: int = 1
    
    # Risk Management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 0.0
    risk_amount: float = 0.0
    
    # Performance
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_minutes: int = 0
    
    # Analysis
    confluences: str = ""  # JSON string of confluences
    ai_analysis: str = ""
    signal_strength: float = 0.0
    timeframe: str = ""
    
    # Market Conditions
    rsi_entry: float = 0.0
    price_change_24h: float = 0.0
    volume_ratio: float = 0.0
    news_sentiment: int = 0
    
    # Status
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    close_reason: str = ""  # TP, SL, MANUAL, TIMEOUT
    notes: str = ""

class TradeJournal:
    """Advanced trade journal with analytics"""
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        print("📓 Trade Journal initialized")
        print(f"💾 Database: {self.db_path}")
    
    def setup_logging(self):
        """Setup logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'ai_trading.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('TradeJournal')
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    leverage INTEGER,
                    stop_loss REAL,
                    take_profit REAL,
                    risk_reward REAL,
                    risk_amount REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    fees REAL,
                    duration_minutes INTEGER,
                    confluences TEXT,
                    ai_analysis TEXT,
                    signal_strength REAL,
                    timeframe TEXT,
                    rsi_entry REAL,
                    price_change_24h REAL,
                    volume_ratio REAL,
                    news_sentiment INTEGER,
                    status TEXT,
                    close_reason TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_pnl REAL,
                    win_rate REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def log_trade_entry(self, trade: TradeRecord) -> bool:
        """Log new trade entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert dataclass to dict
                trade_dict = asdict(trade)
                
                # Convert confluences to JSON string if it's a list
                if isinstance(trade.confluences, list):
                    trade_dict['confluences'] = json.dumps(trade.confluences)
                
                # Insert trade
                columns = ', '.join(trade_dict.keys())
                placeholders = ', '.join(['?' for _ in trade_dict])
                
                conn.execute(
                    f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})",
                    list(trade_dict.values())
                )
            
            self.logger.info(f"✅ Trade logged: {trade.trade_id} {trade.symbol} {trade.direction}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error logging trade: {e}")
            return False
    
    def update_trade_exit(self, trade_id: str, exit_price: float, 
                         close_reason: str, notes: str = "") -> bool:
        """Update trade with exit information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get original trade
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE trade_id = ?", 
                    (trade_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    self.logger.error(f"Trade {trade_id} not found")
                    return False
                
                # Calculate PnL
                entry_price = row[4]  # entry_price column
                direction = row[3]   # direction column
                quantity = row[6]    # quantity column
                leverage = row[7]    # leverage column
                
                if direction == 'BUY':
                    pnl = (exit_price - entry_price) * quantity * leverage
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
                else:  # SELL
                    pnl = (entry_price - exit_price) * quantity * leverage  
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage
                
                # Calculate duration
                entry_time = datetime.fromisoformat(row[1])
                duration = int((datetime.utcnow() - entry_time).total_seconds() / 60)
                
                # Update trade
                conn.execute("""
                    UPDATE trades SET 
                        exit_price = ?, 
                        pnl = ?, 
                        pnl_pct = ?,
                        duration_minutes = ?,
                        status = 'CLOSED',
                        close_reason = ?,
                        notes = ?
                    WHERE trade_id = ?
                """, (exit_price, pnl, pnl_pct, duration, close_reason, notes, trade_id))
            
            self.logger.info(f"✅ Trade closed: {trade_id} PnL: ${pnl:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating trade exit: {e}")
            return False
    
    def get_daily_stats(self, date: str = None) -> Dict[str, Any]:
        """Get daily trading statistics"""
        if not date:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get trades for the day
                cursor = conn.execute("""
                    SELECT * FROM trades 
                    WHERE DATE(timestamp) = ? AND status = 'CLOSED'
                """, (date,))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {
                        'date': date,
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'avg_win': 0.0,
                        'avg_loss': 0.0,
                        'profit_factor': 0.0
                    }
                
                # Calculate statistics
                total_trades = len(trades)
                winning_trades = sum(1 for trade in trades if trade[12] > 0)  # pnl > 0
                losing_trades = total_trades - winning_trades
                total_pnl = sum(trade[12] for trade in trades)  # sum of pnl
                
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                winning_pnls = [trade[12] for trade in trades if trade[12] > 0]
                losing_pnls = [trade[12] for trade in trades if trade[12] < 0]
                
                avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
                avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
                
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                stats = {
                    'date': date,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor
                }
                
                # Save to daily_stats table
                conn.execute("""
                    INSERT OR REPLACE INTO daily_stats 
                    (date, total_trades, winning_trades, losing_trades, total_pnl,
                     win_rate, avg_win, avg_loss, profit_factor)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (date, total_trades, winning_trades, losing_trades, total_pnl,
                      win_rate, avg_win, avg_loss, profit_factor))
                
                return stats
                
        except Exception as e:
            self.logger.error(f"❌ Error getting daily stats: {e}")
            return {}
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for specified period"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM trades 
                    WHERE timestamp >= ? AND status = 'CLOSED'
                    ORDER BY timestamp
                """, (start_date.isoformat(),))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {'error': 'No closed trades found'}
                
                # Calculate comprehensive statistics
                total_trades = len(trades)
                winning_trades = sum(1 for trade in trades if trade[12] > 0)
                losing_trades = total_trades - winning_trades
                
                pnls = [trade[12] for trade in trades]
                total_pnl = sum(pnls)
                
                # Calculate maximum drawdown
                cumulative_pnl = []
                running_total = 0
                for pnl in pnls:
                    running_total += pnl
                    cumulative_pnl.append(running_total)
                
                peak = cumulative_pnl[0]
                max_drawdown = 0
                for value in cumulative_pnl:
                    if value > peak:
                        peak = value
                    drawdown = peak - value
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                # Analyze confluences
                confluence_analysis = self.analyze_confluences(trades)
                
                summary = {
                    'period_days': days,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
                    'max_drawdown': max_drawdown,
                    'best_trade': max(pnls) if pnls else 0,
                    'worst_trade': min(pnls) if pnls else 0,
                    'confluence_analysis': confluence_analysis,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                return summary
                
        except Exception as e:
            self.logger.error(f"❌ Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def analyze_confluences(self, trades: List) -> Dict[str, Any]:
        """Analyze which confluences lead to better trades"""
        confluence_stats = {}
        
        for trade in trades:
            confluences_str = trade[16]  # confluences column
            if confluences_str:
                try:
                    confluences = json.loads(confluences_str)
                    pnl = trade[12]  # pnl column
                    
                    for confluence in confluences:
                        if confluence not in confluence_stats:
                            confluence_stats[confluence] = {
                                'count': 0,
                                'wins': 0,
                                'total_pnl': 0.0
                            }
                        
                        confluence_stats[confluence]['count'] += 1
                        confluence_stats[confluence]['total_pnl'] += pnl
                        
                        if pnl > 0:
                            confluence_stats[confluence]['wins'] += 1
                
                except json.JSONDecodeError:
                    continue
        
        # Calculate win rates for each confluence
        for confluence, stats in confluence_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
            stats['avg_pnl'] = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
        
        # Sort by win rate
        sorted_confluences = dict(sorted(
            confluence_stats.items(), 
            key=lambda x: x[1]['win_rate'], 
            reverse=True
        ))
        
        return sorted_confluences
    
    def export_to_csv(self, filepath: str = None) -> str:
        """Export all trades to CSV"""
        if not filepath:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filepath = f"exports/trades_{timestamp}.csv"
        
        Path(filepath).parent.mkdir(exist_ok=True)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
                df.to_csv(filepath, index=False)
            
            self.logger.info(f"✅ Trades exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting trades: {e}")
            return ""
    
    def print_daily_report(self, date: str = None):
        """Print formatted daily report"""
        stats = self.get_daily_stats(date)
        
        if not stats or stats['total_trades'] == 0:
            print(f"📊 No trades found for {date or 'today'}")
            return
        
        print(f"\n📊 Daily Trading Report - {stats['date']}")
        print("=" * 40)
        print(f"📈 Total Trades: {stats['total_trades']}")
        print(f"✅ Winning Trades: {stats['winning_trades']}")
        print(f"❌ Losing Trades: {stats['losing_trades']}")
        print(f"🎯 Win Rate: {stats['win_rate']:.1f}%")
        print(f"💰 Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"📊 Average Win: ${stats['avg_win']:,.2f}")
        print(f"📉 Average Loss: ${stats['avg_loss']:,.2f}")
        print(f"⚡ Profit Factor: {stats['profit_factor']:.2f}")

def demo_trade_journal():
    """Demo the trade journal system"""
    print("📓 Trade Journal Demo")
    print("=" * 25)
    
    # Initialize journal
    journal = TradeJournal()
    
    # Create sample trades
    sample_trades = [
        TradeRecord(
            trade_id="TRADE_001",
            timestamp=datetime.utcnow().isoformat(),
            symbol="BTC/USDT",
            direction="BUY",
            entry_price=50000.0,
            quantity=0.001,
            leverage=2,
            stop_loss=49000.0,
            take_profit=52000.0,
            risk_reward=2.0,
            confluences=json.dumps(["✅ BOS confirmed", "✅ FVG zone", "✅ RSI oversold"]),
            ai_analysis="Strong bullish setup with multiple confirmations",
            signal_strength=85.0,
            timeframe="1h",
            rsi_entry=28.5,
            status="OPEN"
        ),
        TradeRecord(
            trade_id="TRADE_002", 
            timestamp=(datetime.utcnow() - timedelta(hours=2)).isoformat(),
            symbol="ETH/USDT",
            direction="SELL",
            entry_price=3200.0,
            exit_price=3100.0,
            quantity=0.1,
            leverage=3,
            stop_loss=3250.0,
            take_profit=3050.0,
            risk_reward=3.0,
            pnl=30.0,
            pnl_pct=3.125,
            duration_minutes=120,
            confluences=json.dumps(["✅ Bearish BOS", "✅ EMA resistance", "✅ RSI overbought"]),
            ai_analysis="Clear bearish momentum with resistance rejection",
            signal_strength=78.0,
            timeframe="4h",
            rsi_entry=75.2,
            status="CLOSED",
            close_reason="TP"
        )
    ]
    
    # Log sample trades
    for trade in sample_trades:
        journal.log_trade_entry(trade)
    
    # Update first trade as closed
    journal.update_trade_exit("TRADE_001", 51500.0, "TP", "Good setup executed perfectly")
    
    # Show daily report
    journal.print_daily_report()
    
    # Show performance summary
    summary = journal.get_performance_summary(7)
    if 'error' not in summary:
        print(f"\n📈 7-Day Performance Summary:")
        print("=" * 35)
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"Max Drawdown: ${summary['max_drawdown']:,.2f}")
        
        print(f"\n🎯 Best Performing Confluences:")
        for confluence, stats in list(summary['confluence_analysis'].items())[:3]:
            print(f"   {confluence}: {stats['win_rate']:.1f}% win rate ({stats['count']} trades)")

if __name__ == "__main__":
    demo_trade_journal()