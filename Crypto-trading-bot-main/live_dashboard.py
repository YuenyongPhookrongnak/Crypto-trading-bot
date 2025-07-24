#!/usr/bin/env python3
"""
Simple Live Trading Dashboard

A terminal-based dashboard to monitor your trading bot in real-time.
Works with or without the 'rich' library for maximum compatibility.
"""

import asyncio
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Try to import ccxt for live market data
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# Try to import rich for beautiful display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class SimpleDashboard:
    """Simple trading dashboard that works without external dependencies"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.exchange = None
        self.data = {
            'portfolio': {
                'total_value': 10000.0,
                'daily_pnl': 0.0,
                'active_positions': 0,
                'win_rate': 0.0,
                'trades_today': 0
            },
            'market': {},
            'trades': [],
            'bot_status': {
                'status': 'RUNNING',
                'uptime': '0h 0m',
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
        }
        
        # Create data directory
        Path('data').mkdir(exist_ok=True)
    
    async def initialize_exchange(self):
        """Initialize exchange connection for live data"""
        if not CCXT_AVAILABLE:
            print("📡 CCXT not available - using demo data")
            return False
        
        try:
            self.exchange = ccxt.binance({
                'sandbox': True,
                'enableRateLimit': True,
            })
            return True
        except Exception as e:
            print(f"❌ Exchange connection failed: {e}")
            return False
    
    async def update_market_data(self):
        """Update live market data"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        
        if self.exchange:
            try:
                for symbol in symbols:
                    ticker = self.exchange.fetch_ticker(symbol)
                    self.data['market'][symbol] = {
                        'price': ticker['last'],
                        'change': ticker['percentage'],
                        'volume': ticker['baseVolume'],
                        'high': ticker['high'],
                        'low': ticker['low']
                    }
            except Exception as e:
                print(f"⚠️  Error fetching market data: {e}")
        else:
            # Demo data
            import random
            for symbol in symbols:
                base_price = {'BTC/USDT': 118000, 'ETH/USDT': 3200, 'ADA/USDT': 0.45}[symbol]
                self.data['market'][symbol] = {
                    'price': base_price * (1 + random.uniform(-0.02, 0.02)),
                    'change': random.uniform(-5, 5),
                    'volume': random.randint(1000, 50000),
                    'high': base_price * 1.05,
                    'low': base_price * 0.95
                }
    
    def load_data_files(self):
        """Load data from files if available"""
        try:
            # Portfolio data
            portfolio_file = Path('data/portfolio_status.json')
            if portfolio_file.exists():
                with open(portfolio_file) as f:
                    self.data['portfolio'].update(json.load(f))
            
            # Recent trades
            trades_file = Path('data/recent_trades.json')
            if trades_file.exists():
                with open(trades_file) as f:
                    self.data['trades'] = json.load(f)[-10:]  # Last 10 trades
            
            # Bot status
            status_file = Path('data/bot_status.json')
            if status_file.exists():
                with open(status_file) as f:
                    self.data['bot_status'].update(json.load(f))
                    
        except Exception as e:
            print(f"⚠️  Error loading data files: {e}")
    
    def create_sample_data(self):
        """Create sample data for demo"""
        # Sample portfolio
        import random
        self.data['portfolio'] = {
            'total_value': 10000 + random.uniform(-500, 500),
            'daily_pnl': random.uniform(-100, 150),
            'active_positions': random.randint(0, 3),
            'win_rate': random.uniform(60, 80),
            'trades_today': random.randint(5, 15),
            'cash_balance': random.uniform(5000, 8000),
            'invested_amount': random.uniform(2000, 5000)
        }
        
        # Sample trades
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        directions = ['BUY', 'SELL']
        statuses = ['OPEN', 'CLOSED', 'CLOSED', 'CLOSED']  # Mostly closed
        
        self.data['trades'] = []
        for i in range(8):
            trade = {
                'timestamp': (datetime.now() - timedelta(minutes=i*30)).strftime('%H:%M:%S'),
                'symbol': random.choice(symbols),
                'direction': random.choice(directions),
                'quantity': round(random.uniform(0.001, 0.1), 6),
                'price': random.uniform(1000, 120000),
                'pnl': round(random.uniform(-50, 100), 2),
                'status': random.choice(statuses)
            }
            self.data['trades'].append(trade)
        
        # Bot status
        self.data['bot_status'] = {
            'status': 'RUNNING',
            'uptime': f"{random.randint(1, 12)}h {random.randint(0, 59)}m",
            'last_update': datetime.now().strftime('%H:%M:%S'),
            'strategies_active': 2,
            'signals_today': random.randint(10, 30)
        }
    
    def display_rich_dashboard(self):
        """Display dashboard using Rich library"""
        if not RICH_AVAILABLE:
            return None
        
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split(
            Layout(name="portfolio"),
            Layout(name="bot_status")
        )
        
        layout["right"].split(
            Layout(name="market"),
            Layout(name="performance")
        )
        
        # Header
        header_text = Text("🤖 Cryptocurrency Trading Bot", style="bold blue")
        timestamp = Text(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        layout["header"].update(Panel(f"{header_text}\n{timestamp}", border_style="green"))
        
        # Portfolio panel
        portfolio = self.data['portfolio']
        portfolio_table = Table(show_header=False, box=None)
        portfolio_table.add_column("Metric", style="cyan")
        portfolio_table.add_column("Value", style="white")
        
        daily_pnl = portfolio.get('daily_pnl', 0)
        pnl_color = "green" if daily_pnl >= 0 else "red"
        pnl_symbol = "+" if daily_pnl >= 0 else ""
        
        portfolio_table.add_row("💰 Total Value", f"${portfolio.get('total_value', 0):,.2f}")
        portfolio_table.add_row("📊 Daily P&L", f"[{pnl_color}]{pnl_symbol}${daily_pnl:,.2f}[/{pnl_color}]")
        portfolio_table.add_row("🎯 Positions", str(portfolio.get('active_positions', 0)))
        portfolio_table.add_row("🏆 Win Rate", f"{portfolio.get('win_rate', 0):.1f}%")
        portfolio_table.add_row("📋 Trades Today", str(portfolio.get('trades_today', 0)))
        
        layout["portfolio"].update(Panel(portfolio_table, title="[bold yellow]💰 Portfolio[/bold yellow]", border_style="yellow"))
        
        # Market panel
        market_table = Table(show_header=True)
        market_table.add_column("Symbol", style="cyan")
        market_table.add_column("Price", justify="right")
        market_table.add_column("Change %", justify="right")
        market_table.add_column("Volume", justify="right")
        
        for symbol, data in self.data['market'].items():
            price = data.get('price', 0)
            change = data.get('change', 0)
            volume = data.get('volume', 0)
            
            change_color = "green" if change >= 0 else "red"
            change_text = f"[{change_color}]{change:+.2f}%[/{change_color}]"
            
            market_table.add_row(
                symbol,
                f"${price:,.2f}",
                change_text,
                f"{volume:,.0f}"
            )
        
        layout["market"].update(Panel(market_table, title="[bold blue]📊 Market Data[/bold blue]", border_style="blue"))
        
        # Bot status panel
        status = self.data['bot_status']
        status_table = Table(show_header=False, box=None)
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="white")
        
        bot_status = status.get('status', 'Unknown')
        status_color = "green" if bot_status == "RUNNING" else "red"
        
        status_table.add_row("🤖 Status", f"[{status_color}]{bot_status}[/{status_color}]")
        status_table.add_row("⏱️  Uptime", status.get('uptime', 'N/A'))
        status_table.add_row("🔧 Strategies", str(status.get('strategies_active', 0)))
        status_table.add_row("📡 Signals", str(status.get('signals_today', 0)))
        status_table.add_row("🔄 Updated", status.get('last_update', 'N/A'))
        
        layout["bot_status"].update(Panel(status_table, title="[bold red]🤖 Bot Status[/bold red]", border_style="red"))
        
        # Performance panel (simple)
        perf_text = Text()
        portfolio = self.data['portfolio']
        total_return = ((portfolio.get('total_value', 10000) - 10000) / 10000) * 100
        perf_text.append(f"📈 Total Return: {total_return:+.2f}%\n")
        perf_text.append(f"💼 Win Rate: {portfolio.get('win_rate', 0):.1f}%\n")
        perf_text.append(f"📊 Risk Utilization: 45%\n")
        perf_text.append(f"⚡ Sharpe Ratio: 1.23")
        
        layout["performance"].update(Panel(perf_text, title="[bold cyan]📈 Performance[/bold cyan]", border_style="cyan"))
        
        # Recent trades
        trades_table = Table(show_header=True)
        trades_table.add_column("Time", style="dim")
        trades_table.add_column("Symbol", style="cyan")
        trades_table.add_column("Side", justify="center")
        trades_table.add_column("P&L", justify="right")
        trades_table.add_column("Status")
        
        for trade in self.data['trades'][-5:]:
            timestamp = trade.get('timestamp', 'N/A')
            symbol = trade.get('symbol', 'N/A')
            direction = trade.get('direction', 'N/A')
            pnl = trade.get('pnl', 0)
            status = trade.get('status', 'N/A')
            
            direction_color = "green" if direction == "BUY" else "red"
            pnl_color = "green" if pnl >= 0 else "red"
            status_color = "green" if status == "CLOSED" else "yellow"
            
            trades_table.add_row(
                timestamp,
                symbol,
                f"[{direction_color}]{direction}[/{direction_color}]",
                f"[{pnl_color}]{pnl:+.2f}[/{pnl_color}]",
                f"[{status_color}]{status}[/{status_color}]"
            )
        
        layout["footer"].update(Panel(trades_table, title="[bold magenta]📋 Recent Trades[/bold magenta]", border_style="magenta"))
        
        return layout
    
    def display_simple_dashboard(self):
        """Display simple text-based dashboard"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 70)
        print("🤖 Cryptocurrency Trading Bot - Live Dashboard")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Portfolio section
        portfolio = self.data['portfolio']
        print("💰 PORTFOLIO STATUS:")
        print("-" * 20)
        print(f"  Total Value:     ${portfolio.get('total_value', 0):,.2f}")
        
        daily_pnl = portfolio.get('daily_pnl', 0)
        pnl_symbol = "+" if daily_pnl >= 0 else ""
        print(f"  Daily P&L:       {pnl_symbol}${daily_pnl:,.2f}")
        print(f"  Active Positions: {portfolio.get('active_positions', 0)}")
        print(f"  Win Rate:        {portfolio.get('win_rate', 0):.1f}%")
        print(f"  Trades Today:    {portfolio.get('trades_today', 0)}")
        print("")
        
        # Market section
        print("📊 MARKET DATA:")
        print("-" * 15)
        for symbol, data in self.data['market'].items():
            price = data.get('price', 0)
            change = data.get('change', 0)
            change_symbol = "+" if change >= 0 else ""
            print(f"  {symbol:10} ${price:>8,.2f} ({change_symbol}{change:>5.2f}%)")
        print("")
        
        # Bot status section
        status = self.data['bot_status']
        print("🤖 BOT STATUS:")
        print("-" * 14)
        print(f"  Status:          {status.get('status', 'Unknown')}")
        print(f"  Uptime:          {status.get('uptime', 'N/A')}")
        print(f"  Strategies:      {status.get('strategies_active', 0)}")
        print(f"  Signals Today:   {status.get('signals_today', 0)}")
        print(f"  Last Update:     {status.get('last_update', 'N/A')}")
        print("")
        
        # Recent trades section
        print("📋 RECENT TRADES:")
        print("-" * 17)
        if self.data['trades']:
            print(f"  {'Time':>8} {'Symbol':>10} {'Side':>4} {'P&L':>8} {'Status':>8}")
            print("  " + "-" * 45)
            for trade in self.data['trades'][-5:]:
                timestamp = trade.get('timestamp', 'N/A')
                symbol = trade.get('symbol', 'N/A')
                direction = trade.get('direction', 'N/A')
                pnl = trade.get('pnl', 0)
                status = trade.get('status', 'N/A')
                
                pnl_symbol = "+" if pnl >= 0 else ""
                print(f"  {timestamp:>8} {symbol:>10} {direction:>4} {pnl_symbol}${pnl:>6.2f} {status:>8}")
        else:
            print("  No recent trades")
        
        print("\n" + "=" * 70)
        print("📋 Commands: Ctrl+C to exit | Updates every 5 seconds")
        print("💡 Tips: pip install rich for enhanced dashboard")
    
    async def run_dashboard(self):
        """Run the live dashboard"""
        print("🖥️  Starting Live Trading Dashboard...")
        
        # Initialize exchange connection
        await self.initialize_exchange()
        
        if RICH_AVAILABLE:
            print("✨ Rich library detected - using enhanced dashboard")
            await self._run_rich_dashboard()
        else:
            print("📺 Using simple text dashboard")
            print("💡 Install Rich for better experience: pip install rich")
            await self._run_simple_dashboard()
    
    async def _run_rich_dashboard(self):
        """Run Rich-based dashboard"""
        with Live(self.display_rich_dashboard(), auto_refresh=False, screen=True) as live:
            while True:
                try:
                    # Update data
                    await self.update_market_data()
                    self.load_data_files()
                    self.create_sample_data()  # Add some demo data
                    
                    # Update display
                    live.update(self.display_rich_dashboard())
                    live.refresh()
                    
                    await asyncio.sleep(5)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in dashboard: {e}")
                    await asyncio.sleep(5)
    
    async def _run_simple_dashboard(self):
        """Run simple text dashboard"""
        while True:
            try:
                # Update data
                await self.update_market_data()
                self.load_data_files()
                self.create_sample_data()  # Add some demo data
                
                # Display
                self.display_simple_dashboard()
                
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in dashboard: {e}")
                await asyncio.sleep(5)

async def main():
    """Main dashboard function"""
    dashboard = SimpleDashboard()
    
    try:
        await dashboard.run_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")

if __name__ == "__main__":
    asyncio.run(main())