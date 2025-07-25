#!/usr/bin/env python3
"""
Notification System for AI Trading Bot

Sends alerts via Discord and Telegram for trading signals,
trade execution, and risk management alerts.
"""

import asyncio
import aiohttp
import requests
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class NotificationMessage:
    """Notification message structure"""
    title: str
    message: str
    emoji: str = "📊"
    color: int = 0x00ff00  # Green
    urgent: bool = False

class DiscordNotifier:
    """Discord webhook notifications"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)
    
    async def send_embed(self, title: str, description: str, 
                        color: int = 0x00ff00, fields: List[Dict] = None):
        """Send rich embed message to Discord"""
        if not self.enabled:
            return False
        
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "AI Trading Bot",
                "icon_url": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
            }
        }
        
        if fields:
            embed["fields"] = fields
        
        payload = {
            "embeds": [embed]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 204
        except Exception as e:
            print(f"❌ Discord notification failed: {e}")
            return False
    
    async def send_trading_signal(self, setup: Dict[str, Any]):
        """Send trading signal notification"""
        color = 0x00ff00 if setup['direction'] == 'BUY' else 0xff0000
        emoji = "🟢" if setup['direction'] == 'BUY' else "🔴"
        
        fields = [
            {"name": "💰 Entry Price", "value": f"${setup['entry_price']:,.2f}", "inline": True},
            {"name": "🛑 Stop Loss", "value": f"${setup['stop_loss']:,.2f}", "inline": True},
            {"name": "🎯 Take Profit", "value": f"${setup['take_profit']:,.2f}", "inline": True},
            {"name": "📊 Risk/Reward", "value": f"1:{setup['risk_reward']:.1f}", "inline": True},
            {"name": "🎲 Confidence", "value": f"{setup['confidence']:.1f}%", "inline": True},
            {"name": "⏰ Timeframe", "value": setup['timeframe'], "inline": True}
        ]
        
        if setup.get('confluences'):
            confluences_text = "\n".join([f"• {c}" for c in setup['confluences'][:5]])
            fields.append({"name": "✅ Confluences", "value": confluences_text, "inline": False})
        
        await self.send_embed(
            title=f"{emoji} Trading Signal: {setup['symbol']} {setup['direction']}",
            description=f"AI-detected trading opportunity with {len(setup.get('confluences', []))} confluences",
            color=color,
            fields=fields
        )
    
    async def send_trade_execution(self, trade: Dict[str, Any]):
        """Send trade execution notification"""
        color = 0x0099ff
        
        fields = [
            {"name": "💰 Entry Price", "value": f"${trade['entry_price']:,.2f}", "inline": True},
            {"name": "📊 Position Size", "value": f"{trade['quantity']:.6f}", "inline": True},
            {"name": "📈 Leverage", "value": f"{trade['leverage']}x", "inline": True}
        ]
        
        await self.send_embed(
            title=f"📈 Position Opened: {trade['symbol']} {trade['direction']}",
            description=f"Trade executed successfully",
            color=color,
            fields=fields
        )
    
    async def send_trade_closed(self, trade: Dict[str, Any]):
        """Send trade closure notification"""
        pnl = trade.get('pnl', 0)
        color = 0x00ff00 if pnl > 0 else 0xff0000
        emoji = "✅" if pnl > 0 else "❌"
        
        fields = [
            {"name": "💰 Exit Price", "value": f"${trade['exit_price']:,.2f}", "inline": True},
            {"name": "📊 P&L", "value": f"${pnl:,.2f}", "inline": True},
            {"name": "⏱️ Duration", "value": f"{trade.get('duration_minutes', 0)} min", "inline": True},
            {"name": "🏁 Reason", "value": trade.get('close_reason', 'MANUAL'), "inline": True}
        ]
        
        await self.send_embed(
            title=f"{emoji} Position Closed: {trade['symbol']}",
            description=f"Trade completed with {pnl:+.2f} P&L",
            color=color,
            fields=fields
        )
    
    async def send_risk_alert(self, alert_type: str, message: str):
        """Send risk management alert"""
        await self.send_embed(
            title=f"🚨 Risk Alert: {alert_type}",
            description=message,
            color=0xff0000  # Red
        )

class TelegramNotifier:
    """Telegram bot notifications"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, text: str, parse_mode: str = "Markdown"):
        """Send message to Telegram"""
        if not self.enabled:
            return False
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/sendMessage", json=payload) as response:
                    return response.status == 200
        except Exception as e:
            print(f"❌ Telegram notification failed: {e}")
            return False
    
    async def send_trading_signal(self, setup: Dict[str, Any]):
        """Send trading signal to Telegram"""
        emoji = "🟢" if setup['direction'] == 'BUY' else "🔴"
        
        message = f"""
{emoji} *Trading Signal Detected*

📊 *Symbol:* {setup['symbol']}
📈 *Direction:* {setup['direction']}
💰 *Entry:* ${setup['entry_price']:,.2f}
🛑 *Stop Loss:* ${setup['stop_loss']:,.2f}
🎯 *Take Profit:* ${setup['take_profit']:,.2f}
📊 *R/R:* 1:{setup['risk_reward']:.1f}
🎲 *Confidence:* {setup['confidence']:.1f}%
⏰ *Timeframe:* {setup['timeframe']}

✅ *Confluences:*
{chr(10).join([f"• {c}" for c in setup.get('confluences', [])[:5]])}

🧠 *AI Analysis:*
{setup.get('ai_analysis', 'No analysis available')[:200]}...
        """
        
        await self.send_message(message.strip())
    
    async def send_trade_execution(self, trade: Dict[str, Any]):
        """Send trade execution to Telegram"""
        message = f"""
📈 *Position Opened*

📊 *Symbol:* {trade['symbol']}
📈 *Direction:* {trade['direction']}
💰 *Entry Price:* ${trade['entry_price']:,.2f}
📊 *Size:* {trade['quantity']:.6f}
📈 *Leverage:* {trade['leverage']}x
🕒 *Time:* {datetime.utcnow().strftime('%H:%M:%S')}
        """
        
        await self.send_message(message.strip())
    
    async def send_trade_closed(self, trade: Dict[str, Any]):
        """Send trade closure to Telegram"""
        pnl = trade.get('pnl', 0)
        emoji = "✅" if pnl > 0 else "❌"
        
        message = f"""
{emoji} *Position Closed*

📊 *Symbol:* {trade['symbol']}
💰 *Exit Price:* ${trade['exit_price']:,.2f}
📊 *P&L:* ${pnl:+.2f}
📈 *P&L %:* {trade.get('pnl_pct', 0):+.2f}%
⏱️ *Duration:* {trade.get('duration_minutes', 0)} minutes
🏁 *Reason:* {trade.get('close_reason', 'MANUAL')}
        """
        
        await self.send_message(message.strip())
    
    async def send_daily_summary(self, stats: Dict[str, Any]):
        """Send daily trading summary"""
        message = f"""
📊 *Daily Trading Summary*

📅 *Date:* {stats['date']}
📈 *Total Trades:* {stats['total_trades']}
✅ *Wins:* {stats['winning_trades']}
❌ *Losses:* {stats['losing_trades']}
🎯 *Win Rate:* {stats['win_rate']:.1f}%
💰 *Total P&L:* ${stats['total_pnl']:,.2f}
📊 *Avg Win:* ${stats['avg_win']:,.2f}
📉 *Avg Loss:* ${stats['avg_loss']:,.2f}
⚡ *Profit Factor:* {stats['profit_factor']:.2f}
        """
        
        await self.send_message(message.strip())

class NotificationManager:
    """Centralized notification management"""
    
    def __init__(self):
        # Load configuration
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL', '')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Initialize notifiers
        self.discord = DiscordNotifier(self.discord_webhook)
        self.telegram = TelegramNotifier(self.telegram_token, self.telegram_chat)
        
        print("🔔 Notification Manager initialized")
        print(f"   Discord: {'✅ Active' if self.discord.enabled else '❌ Disabled'}")
        print(f"   Telegram: {'✅ Active' if self.telegram.enabled else '❌ Disabled'}")
    
    async def send_signal_alert(self, setup: Dict[str, Any]):
        """Send trading signal to all enabled channels"""
        tasks = []
        
        if self.discord.enabled:
            tasks.append(self.discord.send_trading_signal(setup))
        
        if self.telegram.enabled:
            tasks.append(self.telegram.send_trading_signal(setup))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_trade_execution(self, trade: Dict[str, Any]):
        """Send trade execution to all enabled channels"""
        tasks = []
        
        if self.discord.enabled:
            tasks.append(self.discord.send_trade_execution(trade))
        
        if self.telegram.enabled:
            tasks.append(self.telegram.send_trade_execution(trade))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_trade_closed(self, trade: Dict[str, Any]):
        """Send trade closure to all enabled channels"""
        tasks = []
        
        if self.discord.enabled:
            tasks.append(self.discord.send_trade_closed(trade))
        
        if self.telegram.enabled:
            tasks.append(self.telegram.send_trade_closed(trade))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_risk_alert(self, alert_type: str, message: str):
        """Send risk alert to all channels"""
        tasks = []
        
        if self.discord.enabled:
            tasks.append(self.discord.send_risk_alert(alert_type, message))
        
        if self.telegram.enabled:
            tasks.append(self.telegram.send_message(f"🚨 *Risk Alert: {alert_type}*\n\n{message}"))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_daily_summary(self, stats: Dict[str, Any]):
        """Send daily summary to Telegram"""
        if self.telegram.enabled:
            await self.telegram.send_daily_summary(stats)

async def demo_notifications():
    """Demo notification system"""
    print("🔔 Notification System Demo")
    print("=" * 30)
    
    # Initialize manager
    manager = NotificationManager()
    
    if not (manager.discord.enabled or manager.telegram.enabled):
        print("⚠️  No notification channels configured")
        print("Set DISCORD_WEBHOOK_URL and/or TELEGRAM_BOT_TOKEN in .env")
        return
    
    # Demo trading signal
    demo_setup = {
        'symbol': 'BTC/USDT',
        'direction': 'BUY',
        'entry_price': 50000.0,
        'stop_loss': 49000.0,
        'take_profit': 52000.0,
        'risk_reward': 2.0,
        'confidence': 85.0,
        'timeframe': '1h',
        'confluences': [
            '✅ BOS bullish confirmed',
            '✅ FVG zone entry',
            '✅ RSI oversold reversal',
            '✅ EMA 20/50 alignment',
            '✅ Support level retest'
        ],
        'ai_analysis': 'Strong bullish setup with multiple technical confirmations. Price showing clear break of structure with FVG entry opportunity.'
    }
    
    print("📤 Sending demo trading signal...")
    await manager.send_signal_alert(demo_setup)
    
    # Demo trade execution
    demo_trade = {
        'symbol': 'BTC/USDT',
        'direction': 'BUY',
        'entry_price': 50000.0,
        'quantity': 0.001,
        'leverage': 2
    }
    
    print("📤 Sending demo trade execution...")
    await manager.send_trade_execution(demo_trade)
    
    # Demo trade closure
    demo_closure = {
        'symbol': 'BTC/USDT',
        'exit_price': 51500.0,
        'pnl': 3.0,
        'pnl_pct': 6.0,
        'duration_minutes': 45,
        'close_reason': 'TP'
    }
    
    print("📤 Sending demo trade closure...")
    await manager.send_trade_closed(demo_closure)
    
    print("✅ Demo notifications sent!")

if __name__ == "__main__":
    asyncio.run(demo_notifications())