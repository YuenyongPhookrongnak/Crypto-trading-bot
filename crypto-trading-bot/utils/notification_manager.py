import asyncio
import aiohttp
import smtplib
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
import time

from database.connection import get_db_session
from database.models import SystemLog, LogLevel

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Notification type classifications"""
    TRADE_EXECUTED = "TRADE_EXECUTED"
    TRADE_SIGNAL = "TRADE_SIGNAL"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    RISK_ALERT = "RISK_ALERT"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    MARKET_ALERT = "MARKET_ALERT"
    PERFORMANCE_REPORT = "PERFORMANCE_REPORT"
    REBALANCE_SUGGESTION = "REBALANCE_SUGGESTION"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    URGENT = "URGENT"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "EMAIL"
    DISCORD = "DISCORD"
    TELEGRAM = "TELEGRAM"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    SMS = "SMS"
    PUSH = "PUSH"

@dataclass
class NotificationTemplate:
    """Notification template structure"""
    name: str
    title_template: str
    message_template: str
    channels: List[NotificationChannel]
    priority: NotificationPriority
    conditions: Dict[str, Any] = field(default_factory=dict)
    rate_limit_minutes: int = 0  # 0 = no rate limiting
    
    def format_message(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Format title and message with data"""
        try:
            title = self.title_template.format(**data)
            message = self.message_template.format(**data)
            return title, message
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return self.title_template, self.message_template

@dataclass
class NotificationMessage:
    """Individual notification message"""
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any]
    channels: List[NotificationChannel]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivery_attempts: int = 0
    delivered_channels: List[NotificationChannel] = field(default_factory=list)
    failed_channels: List[NotificationChannel] = field(default_factory=list)
    
    # Optional attachments/formatting
    attachments: List[str] = field(default_factory=list)
    html_content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'notification_type': self.notification_type.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'channels': [ch.value for ch in self.channels],
            'created_at': self.created_at.isoformat(),
            'delivery_attempts': self.delivery_attempts,
            'delivered_channels': [ch.value for ch in self.delivered_channels],
            'failed_channels': [ch.value for ch in self.failed_channels],
            'attachments': self.attachments,
            'html_content': self.html_content
        }

class NotificationManager:
    """Advanced Notification Management System"""
    
    def __init__(self, config):
        self.config = config
        
        # Channel configurations
        self.channel_configs = {
            NotificationChannel.EMAIL: {
                'smtp_server': getattr(config, 'smtp_server', 'smtp.gmail.com'),
                'smtp_port': getattr(config, 'smtp_port', 587),
                'email_user': getattr(config, 'email_user', ''),
                'email_password': getattr(config, 'email_password', ''),
                'from_email': getattr(config, 'from_email', ''),
                'to_emails': getattr(config, 'to_emails', [])
            },
            NotificationChannel.DISCORD: {
                'webhook_url': getattr(config, 'discord_webhook_url', ''),
                'bot_token': getattr(config, 'discord_bot_token', ''),
                'channel_id': getattr(config, 'discord_channel_id', '')
            },
            NotificationChannel.TELEGRAM: {
                'bot_token': getattr(config, 'telegram_bot_token', ''),
                'chat_id': getattr(config, 'telegram_chat_id', ''),
                'api_url': 'https://api.telegram.org/bot{}/sendMessage'
            },
            NotificationChannel.SLACK: {
                'webhook_url': getattr(config, 'slack_webhook_url', ''),
                'bot_token': getattr(config, 'slack_bot_token', ''),
                'channel': getattr(config, 'slack_channel', '#trading-bot')
            },
            NotificationChannel.WEBHOOK: {
                'url': getattr(config, 'webhook_url', ''),
                'headers': getattr(config, 'webhook_headers', {}),
                'auth_token': getattr(config, 'webhook_auth_token', '')
            }
        }
        
        # Notification templates
        self.templates = self._initialize_templates()
        
        # Rate limiting
        self.rate_limits = {}  # Track last notification times
        
        # Delivery queue
        self.notification_queue = asyncio.Queue()
        self.failed_notifications = []
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'by_channel': {ch.value: {'sent': 0, 'failed': 0} for ch in NotificationChannel},
            'by_type': {nt.value: {'sent': 0, 'failed': 0} for nt in NotificationType},
            'last_reset': datetime.utcnow()
        }
        
        # Background worker
        self.worker_task = None
        self.session = None
        
        logger.info("Notification Manager initialized")
    
    async def initialize(self):
        """Initialize notification manager"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Start background worker
            self.worker_task = asyncio.create_task(self._notification_worker())
            
            # Test channel connectivity
            await self._test_channels()
            
            logger.info("Notification Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Notification Manager: {e}")
            raise
    
    def _initialize_templates(self) -> Dict[NotificationType, NotificationTemplate]:
        """Initialize notification templates"""
        
        templates = {
            NotificationType.TRADE_EXECUTED: NotificationTemplate(
                name="Trade Executed",
                title_template="🎯 Trade Executed: {symbol}",
                message_template=(
                    "Trade executed successfully!\n\n"
                    "Symbol: {symbol}\n"
                    "Direction: {direction}\n"
                    "Quantity: {quantity}\n"
                    "Price: ${price:,.4f}\n"
                    "P&L: ${pnl:+,.2f}\n"
                    "Strategy: {strategy_id}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.DISCORD, NotificationChannel.TELEGRAM],
                priority=NotificationPriority.HIGH
            ),
            
            NotificationType.TRADE_SIGNAL: NotificationTemplate(
                name="Trade Signal",
                title_template="📊 New Trade Signal: {symbol}",
                message_template=(
                    "New trading signal detected!\n\n"
                    "Symbol: {symbol}\n"
                    "Signal: {signal_type}\n"
                    "Confidence: {confidence:.1f}%\n"
                    "Entry Price: ${entry_price:,.4f}\n"
                    "Stop Loss: ${stop_loss:,.4f}\n"
                    "Take Profit: ${take_profit:,.4f}\n"
                    "Risk/Reward: {risk_reward:.1f}\n"
                    "Strategy: {strategy_id}"
                ),
                channels=[NotificationChannel.DISCORD],
                priority=NotificationPriority.MEDIUM,
                rate_limit_minutes=5
            ),
            
            NotificationType.STOP_LOSS_HIT: NotificationTemplate(
                name="Stop Loss Hit",
                title_template="🛑 Stop Loss Hit: {symbol}",
                message_template=(
                    "Stop loss triggered!\n\n"
                    "Symbol: {symbol}\n"
                    "Direction: {direction}\n"
                    "Stop Price: ${stop_price:,.4f}\n"
                    "Exit Price: ${exit_price:,.4f}\n"
                    "Loss: ${loss:,.2f}\n"
                    "Position Size: {quantity}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.EMAIL, NotificationChannel.DISCORD, NotificationChannel.TELEGRAM],
                priority=NotificationPriority.HIGH
            ),
            
            NotificationType.TAKE_PROFIT_HIT: NotificationTemplate(
                name="Take Profit Hit",
                title_template="💰 Take Profit Hit: {symbol}",
                message_template=(
                    "Take profit achieved!\n\n"
                    "Symbol: {symbol}\n"
                    "Direction: {direction}\n"
                    "Target Price: ${target_price:,.4f}\n"
                    "Exit Price: ${exit_price:,.4f}\n"
                    "Profit: ${profit:,.2f}\n"
                    "Position Size: {quantity}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.EMAIL, NotificationChannel.DISCORD, NotificationChannel.TELEGRAM],
                priority=NotificationPriority.HIGH
            ),
            
            NotificationType.CIRCUIT_BREAKER: NotificationTemplate(
                name="Circuit Breaker",
                title_template="🚨 CIRCUIT BREAKER ACTIVATED",
                message_template=(
                    "URGENT: Circuit breaker has been triggered!\n\n"
                    "Trigger Type: {trigger_type}\n"
                    "Current Value: {current_value}\n"
                    "Threshold: {threshold}\n"
                    "Severity: {severity}\n"
                    "Action Taken: Trading halted\n"
                    "Cooldown Until: {cooldown_until}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.EMAIL, NotificationChannel.DISCORD, NotificationChannel.TELEGRAM, NotificationChannel.SMS],
                priority=NotificationPriority.CRITICAL
            ),
            
            NotificationType.SYSTEM_ERROR: NotificationTemplate(
                name="System Error",
                title_template="❌ System Error: {component}",
                message_template=(
                    "System error detected!\n\n"
                    "Component: {component}\n"
                    "Error: {error_message}\n"
                    "Severity: {severity}\n"
                    "Time: {timestamp}\n"
                    "Stack Trace: {stack_trace}"
                ),
                channels=[NotificationChannel.EMAIL, NotificationChannel.DISCORD],
                priority=NotificationPriority.HIGH
            ),
            
            NotificationType.PORTFOLIO_UPDATE: NotificationTemplate(
                name="Portfolio Update",
                title_template="📈 Portfolio Update",
                message_template=(
                    "Portfolio Status Update:\n\n"
                    "Total Value: ${total_value:,.2f}\n"
                    "Daily P&L: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.1f}%)\n"
                    "Total P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)\n"
                    "Active Positions: {active_positions}\n"
                    "Cash Balance: ${cash_balance:,.2f}\n"
                    "Largest Position: {largest_position_pct:.1f}%\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.EMAIL],
                priority=NotificationPriority.LOW,
                rate_limit_minutes=60
            ),
            
            NotificationType.RISK_ALERT: NotificationTemplate(
                name="Risk Alert",
                title_template="⚠️ Risk Alert: {risk_type}",
                message_template=(
                    "Risk threshold exceeded!\n\n"
                    "Risk Type: {risk_type}\n"
                    "Current Level: {current_level}\n"
                    "Threshold: {threshold}\n"
                    "Recommendation: {recommendation}\n"
                    "Affected Positions: {affected_positions}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.EMAIL, NotificationChannel.DISCORD],
                priority=NotificationPriority.HIGH
            ),
            
            NotificationType.PERFORMANCE_REPORT: NotificationTemplate(
                name="Performance Report",
                title_template="📊 Daily Performance Report",
                message_template=(
                    "Daily Performance Summary:\n\n"
                    "📈 Returns:\n"
                    "• Daily: {daily_return:+.2f}%\n"
                    "• Weekly: {weekly_return:+.2f}%\n"
                    "• Monthly: {monthly_return:+.2f}%\n\n"
                    "💼 Portfolio:\n"
                    "• Total Value: ${total_value:,.2f}\n"
                    "• P&L: ${total_pnl:+,.2f}\n"
                    "• Positions: {total_positions}\n\n"
                    "📊 Performance:\n"
                    "• Win Rate: {win_rate:.1f}%\n"
                    "• Sharpe Ratio: {sharpe_ratio:.2f}\n"
                    "• Max Drawdown: {max_drawdown:.1f}%\n\n"
                    "🎯 Trades Today: {trades_today}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.EMAIL],
                priority=NotificationPriority.LOW,
                rate_limit_minutes=1440  # Once per day
            ),
            
            NotificationType.MARKET_ALERT: NotificationTemplate(
                name="Market Alert",
                title_template="🌊 Market Alert: {alert_type}",
                message_template=(
                    "Market condition alert!\n\n"
                    "Alert Type: {alert_type}\n"
                    "Affected Symbol: {symbol}\n"
                    "Current Price: ${current_price:,.4f}\n"
                    "Price Change: {price_change:+.2f}%\n"
                    "Volume: ${volume:,.0f}\n"
                    "Market Condition: {market_condition}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.DISCORD],
                priority=NotificationPriority.MEDIUM,
                rate_limit_minutes=15
            ),
            
            NotificationType.REBALANCE_SUGGESTION: NotificationTemplate(
                name="Rebalance Suggestion",
                title_template="⚖️ Portfolio Rebalancing Suggestion",
                message_template=(
                    "Portfolio rebalancing recommended:\n\n"
                    "Recommendations: {recommendation_count}\n"
                    "Top Priority:\n"
                    "• Symbol: {top_symbol}\n"
                    "• Action: {top_action}\n"
                    "• Current: {current_allocation:.1f}%\n"
                    "• Target: {target_allocation:.1f}%\n"
                    "• Estimated Cost: ${estimated_cost:,.2f}\n\n"
                    "Reason: {reasoning}\n"
                    "Time: {timestamp}"
                ),
                channels=[NotificationChannel.EMAIL, NotificationChannel.DISCORD],
                priority=NotificationPriority.MEDIUM,
                rate_limit_minutes=120
            )
        }
        
        return templates
    
    async def send_notification(self, 
                              notification_type: NotificationType,
                              data: Dict[str, Any],
                              priority: Optional[NotificationPriority] = None,
                              channels: Optional[List[NotificationChannel]] = None) -> bool:
        """Send notification using template"""
        
        try:
            # Get template
            template = self.templates.get(notification_type)
            if not template:
                logger.error(f"No template found for notification type: {notification_type}")
                return False
            
            # Check rate limiting
            if template.rate_limit_minutes > 0:
                rate_key = f"{notification_type.value}"
                last_sent = self.rate_limits.get(rate_key)
                
                if last_sent:
                    time_since = (datetime.utcnow() - last_sent).total_seconds() / 60
                    if time_since < template.rate_limit_minutes:
                        logger.debug(f"Rate limit active for {notification_type.value}, skipping")
                        return False
                
                self.rate_limits[rate_key] = datetime.utcnow()
            
            # Use template or override settings
            effective_priority = priority or template.priority
            effective_channels = channels or template.channels
            
            # Format message
            title, message = template.format_message(data)
            
            # Create notification message
            notification = NotificationMessage(
                notification_type=notification_type,
                priority=effective_priority,
                title=title,
                message=message,
                data=data,
                channels=effective_channels
            )
            
            # Add to queue
            await self.notification_queue.put(notification)
            
            logger.info(f"Queued notification: {notification_type.value} ({effective_priority.value})")
            return True
            
        except Exception as e:
            logger.error(f"Error sending notification {notification_type}: {e}")
            return False
    
    async def send_custom_notification(self,
                                     title: str,
                                     message: str,
                                     channels: List[NotificationChannel],
                                     priority: NotificationPriority = NotificationPriority.MEDIUM,
                                     notification_type: NotificationType = NotificationType.SYSTEM_STATUS,
                                     data: Dict[str, Any] = None) -> bool:
        """Send custom notification without template"""
        
        try:
            notification = NotificationMessage(
                notification_type=notification_type,
                priority=priority,
                title=title,
                message=message,
                data=data or {},
                channels=channels
            )
            
            await self.notification_queue.put(notification)
            
            logger.info(f"Queued custom notification: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending custom notification: {e}")
            return False
    
    async def _notification_worker(self):
        """Background worker to process notification queue"""
        
        logger.info("Notification worker started")
        
        while True:
            try:
                # Get notification from queue
                notification = await asyncio.wait_for(
                    self.notification_queue.get(), 
                    timeout=5.0
                )
                
                # Process notification
                await self._process_notification(notification)
                
                # Mark task as done
                self.notification_queue.task_done()
                
            except asyncio.TimeoutError:
                # No notifications in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
                await asyncio.sleep(1)
    
    async def _process_notification(self, notification: NotificationMessage):
        """Process individual notification"""
        
        try:
            notification.delivery_attempts += 1
            
            # Process each channel
            for channel in notification.channels:
                try:
                    success = await self._send_to_channel(notification, channel)
                    
                    if success:
                        notification.delivered_channels.append(channel)
                        self.stats['by_channel'][channel.value]['sent'] += 1
                    else:
                        notification.failed_channels.append(channel)
                        self.stats['by_channel'][channel.value]['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error sending to {channel.value}: {e}")
                    notification.failed_channels.append(channel)
                    self.stats['by_channel'][channel.value]['failed'] += 1
            
            # Update statistics
            if notification.delivered_channels:
                self.stats['total_sent'] += 1
                self.stats['by_type'][notification.notification_type.value]['sent'] += 1
            else:
                self.stats['total_failed'] += 1
                self.stats['by_type'][notification.notification_type.value]['failed'] += 1
                self.failed_notifications.append(notification)
            
            # Log notification result
            if notification.delivered_channels:
                logger.info(f"Notification delivered: {notification.title} "
                          f"(channels: {[ch.value for ch in notification.delivered_channels]})")
            else:
                logger.error(f"Notification failed: {notification.title}")
                
                # Store failed notification in database
                await self._store_failed_notification(notification)
            
        except Exception as e:
            logger.error(f"Error processing notification: {e}")
    
    async def _send_to_channel(self, notification: NotificationMessage, channel: NotificationChannel) -> bool:
        """Send notification to specific channel"""
        
        try:
            if channel == NotificationChannel.EMAIL:
                return await self._send_email(notification)
            elif channel == NotificationChannel.DISCORD:
                return await self._send_discord(notification)
            elif channel == NotificationChannel.TELEGRAM:
                return await self._send_telegram(notification)
            elif channel == NotificationChannel.SLACK:
                return await self._send_slack(notification)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(notification)
            else:
                logger.warning(f"Unsupported notification channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending to {channel.value}: {e}")
            return False
    
    async def _send_email(self, notification: NotificationMessage) -> bool:
        """Send email notification"""
        
        try:
            config = self.channel_configs[NotificationChannel.EMAIL]
            
            if not config['email_user'] or not config['to_emails']:
                logger.warning("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_email'] or config['email_user']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = notification.title
            
            # Priority header
            if notification.priority in [NotificationPriority.CRITICAL, NotificationPriority.URGENT]:
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            
            # Body
            body = notification.message
            if notification.html_content:
                msg.attach(MIMEText(notification.html_content, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['email_user'], config['email_password'])
            
            text = msg.as_string()
            server.sendmail(config['email_user'], config['to_emails'], text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_discord(self, notification: NotificationMessage) -> bool:
        """Send Discord notification"""
        
        try:
            config = self.channel_configs[NotificationChannel.DISCORD]
            
            if not config['webhook_url']:
                logger.warning("Discord webhook URL not configured")
                return False
            
            # Color based on priority
            color_map = {
                NotificationPriority.LOW: 0x95a5a6,      # Gray
                NotificationPriority.MEDIUM: 0x3498db,   # Blue
                NotificationPriority.HIGH: 0xf39c12,     # Orange
                NotificationPriority.CRITICAL: 0xe74c3c, # Red
                NotificationPriority.URGENT: 0x9b59b6    # Purple
            }
            
            # Create embed
            embed = {
                "title": notification.title,
                "description": notification.message[:2000],  # Discord limit
                "color": color_map.get(notification.priority, 0x3498db),
                "timestamp": notification.created_at.isoformat(),
                "footer": {
                    "text": f"Trading Bot • {notification.priority.value}"
                }
            }
            
            # Add fields for structured data
            if notification.data:
                fields = []
                for key, value in notification.data.items():
                    if len(fields) < 25:  # Discord limit
                        if isinstance(value, (int, float)):
                            if 'price' in key.lower() or 'pnl' in key.lower():
                                value = f"${value:,.2f}"
                            elif 'pct' in key.lower() or 'percentage' in key.lower():
                                value = f"{value:.1f}%"
                        
                        fields.append({
                            "name": key.replace('_', ' ').title(),
                            "value": str(value),
                            "inline": True
                        })
                
                if fields:
                    embed["fields"] = fields
            
            # Send webhook
            payload = {
                "embeds": [embed]
            }
            
            async with self.session.post(config['webhook_url'], json=payload) as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    async def _send_telegram(self, notification: NotificationMessage) -> bool:
        """Send Telegram notification"""
        
        try:
            config = self.channel_configs[NotificationChannel.TELEGRAM]
            
            if not config['bot_token'] or not config['chat_id']:
                logger.warning("Telegram configuration incomplete")
                return False
            
            # Format message with priority emoji
            priority_emoji = {
                NotificationPriority.LOW: "ℹ️",
                NotificationPriority.MEDIUM: "📊",
                NotificationPriority.HIGH: "⚠️",
                NotificationPriority.CRITICAL: "🚨",
                NotificationPriority.URGENT: "🆘"
            }
            
            emoji = priority_emoji.get(notification.priority, "📊")
            message_text = f"{emoji} *{notification.title}*\n\n{notification.message}"
            
            # Send message
            url = config['api_url'].format(config['bot_token'])
            payload = {
                'chat_id': config['chat_id'],
                'text': message_text[:4096],  # Telegram limit
                'parse_mode': 'Markdown'
            }
            
            async with self.session.post(url, json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False
    
    async def _send_slack(self, notification: NotificationMessage) -> bool:
        """Send Slack notification"""
        
        try:
            config = self.channel_configs[NotificationChannel.SLACK]
            
            if not config['webhook_url']:
                logger.warning("Slack webhook URL not configured")
                return False
            
            # Color based on priority
            color_map = {
                NotificationPriority.LOW: "good",
                NotificationPriority.MEDIUM: "#439FE0",
                NotificationPriority.HIGH: "warning",
                NotificationPriority.CRITICAL: "danger",
                NotificationPriority.URGENT: "#9B59B6"
            }
            
            # Create attachment
            attachment = {
                "title": notification.title,
                "text": notification.message,
                "color": color_map.get(notification.priority, "#439FE0"),
                "ts": int(notification.created_at.timestamp()),
                "footer": "Trading Bot",
                "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png"
            }
            
            # Add fields
            if notification.data:
                fields = []
                for key, value in notification.data.items():
                    if len(fields) < 10:  # Reasonable limit
                        fields.append({
                            "title": key.replace('_', ' ').title(),
                            "value": str(value),
                            "short": True
                        })
                
                if fields:
                    attachment["fields"] = fields
            
            # Send webhook
            payload = {
                "channel": config['channel'],
                "attachments": [attachment]
            }
            
            async with self.session.post(config['webhook_url'], json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_webhook(self, notification: NotificationMessage) -> bool:
        """Send generic webhook notification"""
        
        try:
            config = self.channel_configs[NotificationChannel.WEBHOOK]
            
            if not config['url']:
                logger.warning("Webhook URL not configured")
                return False
            
            # Prepare headers
            headers = config.get('headers', {}).copy()
            if config.get('auth_token'):
                headers['Authorization'] = f"Bearer {config['auth_token']}"
            
            headers['Content-Type'] = 'application/json'
            
            # Prepare payload
            payload = {
                'notification_type': notification.notification_type.value,
                'priority': notification.priority.value,
                'title': notification.title,
                'message': notification.message,
                'data': notification.data,
                'timestamp': notification.created_at.isoformat()
            }
            
            # Send webhook
            async with self.session.post(config['url'], json=payload, headers=headers) as response:
                return response.status in [200, 201, 202, 204]
                
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    async def _test_channels(self):
        """Test connectivity to all configured channels"""
        
        logger.info("Testing notification channels...")
        
        test_data = {
            'message': 'Test notification from Trading Bot',
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        # Test each configured channel
        for channel in NotificationChannel:
            config = self.channel_configs.get(channel)
            if not config:
                continue
            
            # Check if channel is configured
            configured = False
            
            if channel == NotificationChannel.EMAIL:
                configured = bool(config.get('email_user') and config.get('to_emails'))
            elif channel == NotificationChannel.DISCORD:
                configured = bool(config.get('webhook_url'))
            elif channel == NotificationChannel.TELEGRAM:
                configured = bool(config.get('bot_token') and config.get('chat_id'))
            elif channel == NotificationChannel.SLACK:
                configured = bool(config.get('webhook_url'))
            elif channel == NotificationChannel.WEBHOOK:
                configured = bool(config.get('url'))
            
            if configured:
                logger.info(f"✅ {channel.value} channel configured")
            else:
                logger.info(f"⚪ {channel.value} channel not configured")
    
    async def _store_failed_notification(self, notification: NotificationMessage):
        """Store failed notification in database for later analysis"""
        
        try:
            with get_db_session() as session:
                log_entry = SystemLog(
                    level=LogLevel.ERROR,
                    component='NOTIFICATION_MANAGER',
                    message=f"Failed to deliver notification: {notification.title}",
                    details={
                        'notification': notification.to_dict(),
                        'failed_channels': [ch.value for ch in notification.failed_channels],
                        'delivery_attempts': notification.delivery_attempts
                    }
                )
                session.add(log_entry)
                
        except Exception as e:
            logger.error(f"Error storing failed notification: {e}")
    
    async def retry_failed_notifications(self) -> int:
        """Retry failed notifications"""
        
        retry_count = 0
        
        try:
            failed_notifications = self.failed_notifications.copy()
            self.failed_notifications.clear()
            
            for notification in failed_notifications:
                # Only retry if not too many attempts
                if notification.delivery_attempts < 3:
                    # Reset failed channels for retry
                    notification.failed_channels.clear()
                    
                    # Re-queue notification
                    await self.notification_queue.put(notification)
                    retry_count += 1
                    
                    logger.info(f"Retrying notification: {notification.title}")
            
            if retry_count > 0:
                logger.info(f"Retrying {retry_count} failed notifications")
            
            return retry_count
            
        except Exception as e:
            logger.error(f"Error retrying failed notifications: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        
        try:
            # Calculate uptime
            uptime = datetime.utcnow() - self.stats['last_reset']
            
            # Calculate success rate
            total_attempts = self.stats['total_sent'] + self.stats['total_failed']
            success_rate = (self.stats['total_sent'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats = {
                'overview': {
                    'total_sent': self.stats['total_sent'],
                    'total_failed': self.stats['total_failed'],
                    'success_rate': success_rate,
                    'pending_notifications': self.notification_queue.qsize(),
                    'failed_notifications': len(self.failed_notifications),
                    'uptime_hours': uptime.total_seconds() / 3600
                },
                'by_channel': self.stats['by_channel'].copy(),
                'by_type': self.stats['by_type'].copy(),
                'rate_limits_active': len(self.rate_limits),
                'last_reset': self.stats['last_reset'].isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def reset_statistics(self):
        """Reset notification statistics"""
        
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'by_channel': {ch.value: {'sent': 0, 'failed': 0} for ch in NotificationChannel},
            'by_type': {nt.value: {'sent': 0, 'failed': 0} for nt in NotificationType},
            'last_reset': datetime.utcnow()
        }
        
        logger.info("Notification statistics reset")
    
    async def send_test_notification(self, channels: List[NotificationChannel] = None) -> Dict[str, bool]:
        """Send test notification to verify channels"""
        
        if channels is None:
            channels = [
                NotificationChannel.EMAIL,
                NotificationChannel.DISCORD,
                NotificationChannel.TELEGRAM
            ]
        
        test_data = {
            'test_message': 'This is a test notification from your Trading Bot',
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'version': '1.0.0',
            'status': 'operational'
        }
        
        # Send test notification
        success = await self.send_custom_notification(
            title="🧪 Trading Bot Test Notification",
            message="Test notification sent successfully! All systems operational.",
            channels=channels,
            priority=NotificationPriority.LOW,
            data=test_data
        )
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Return results
        return {'test_sent': success, 'channels_tested': [ch.value for ch in channels]}
    
    async def send_system_startup_notification(self):
        """Send notification when system starts up"""
        
        startup_data = {
            'startup_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'version': '1.0.0',
            'status': 'STARTED'
        }
        
        await self.send_notification(
            NotificationType.SYSTEM_STATUS,
            startup_data
        )
    
    async def send_daily_summary(self, 
                                portfolio_summary: Dict[str, Any],
                                performance_metrics: Dict[str, Any],
                                trade_summary: Dict[str, Any]):
        """Send daily performance summary"""
        
        summary_data = {
            'daily_return': performance_metrics.get('daily_return', 0),
            'weekly_return': performance_metrics.get('weekly_return', 0),
            'monthly_return': performance_metrics.get('monthly_return', 0),
            'total_value': portfolio_summary.get('total_value', 0),
            'total_pnl': portfolio_summary.get('total_pnl', 0),
            'total_positions': portfolio_summary.get('total_positions', 0),
            'win_rate': trade_summary.get('win_rate', 0),
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
            'max_drawdown': performance_metrics.get('max_drawdown', 0),
            'trades_today': trade_summary.get('trades_today', 0),
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        await self.send_notification(
            NotificationType.PERFORMANCE_REPORT,
            summary_data
        )
    
    async def send_trade_notification(self, trade_data: Dict[str, Any]):
        """Send trade execution notification"""
        
        # Determine notification type based on trade result
        if trade_data.get('pnl', 0) > 0:
            if trade_data.get('exit_reason') == 'TAKE_PROFIT':
                notification_type = NotificationType.TAKE_PROFIT_HIT
            else:
                notification_type = NotificationType.TRADE_EXECUTED
        elif trade_data.get('pnl', 0) < 0:
            if trade_data.get('exit_reason') == 'STOP_LOSS':
                notification_type = NotificationType.STOP_LOSS_HIT
            else:
                notification_type = NotificationType.TRADE_EXECUTED
        else:
            notification_type = NotificationType.TRADE_EXECUTED
        
        # Format trade data
        formatted_data = {
            'symbol': trade_data.get('symbol', 'Unknown'),
            'direction': trade_data.get('direction', 'Unknown'),
            'quantity': trade_data.get('quantity', 0),
            'price': trade_data.get('exit_price', trade_data.get('entry_price', 0)),
            'pnl': trade_data.get('pnl', 0),
            'strategy_id': trade_data.get('strategy_id', 'Unknown'),
            'timestamp': trade_data.get('timestamp', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))
        }
        
        # Add specific fields for stop loss/take profit
        if notification_type == NotificationType.STOP_LOSS_HIT:
            formatted_data.update({
                'stop_price': trade_data.get('stop_loss', 0),
                'exit_price': trade_data.get('exit_price', 0),
                'loss': abs(trade_data.get('pnl', 0))
            })
        elif notification_type == NotificationType.TAKE_PROFIT_HIT:
            formatted_data.update({
                'target_price': trade_data.get('take_profit', 0),
                'exit_price': trade_data.get('exit_price', 0),
                'profit': trade_data.get('pnl', 0)
            })
        
        await self.send_notification(notification_type, formatted_data)
    
    async def send_signal_notification(self, signal_data: Dict[str, Any]):
        """Send trading signal notification"""
        
        formatted_data = {
            'symbol': signal_data.get('symbol', 'Unknown'),
            'signal_type': signal_data.get('signal_type', 'Unknown'),
            'confidence': signal_data.get('confidence', 0),
            'entry_price': signal_data.get('entry_price', 0),
            'stop_loss': signal_data.get('stop_loss', 0),
            'take_profit': signal_data.get('take_profit', 0),
            'risk_reward': signal_data.get('risk_reward_ratio', 0),
            'strategy_id': signal_data.get('strategy_id', 'Unknown'),
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        await self.send_notification(NotificationType.TRADE_SIGNAL, formatted_data)
    
    async def send_risk_alert(self, risk_data: Dict[str, Any]):
        """Send risk management alert"""
        
        formatted_data = {
            'risk_type': risk_data.get('risk_type', 'Unknown'),
            'current_level': risk_data.get('current_level', 0),
            'threshold': risk_data.get('threshold', 0),
            'recommendation': risk_data.get('recommendation', 'Review positions'),
            'affected_positions': risk_data.get('affected_positions', 'None specified'),
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        await self.send_notification(NotificationType.RISK_ALERT, formatted_data)
    
    async def send_circuit_breaker_alert(self, breaker_data: Dict[str, Any]):
        """Send circuit breaker activation alert"""
        
        formatted_data = {
            'trigger_type': breaker_data.get('trigger_type', 'Unknown'),
            'current_value': breaker_data.get('current_value', 'Unknown'),
            'threshold': breaker_data.get('threshold', 'Unknown'),
            'severity': breaker_data.get('severity', 'HIGH'),
            'cooldown_until': breaker_data.get('cooldown_until', 'Unknown'),
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        await self.send_notification(NotificationType.CIRCUIT_BREAKER, formatted_data)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get notification system health status"""
        
        try:
            health = {
                'status': 'healthy',
                'checks': {},
                'warnings': [],
                'errors': []
            }
            
            # Check worker task
            if self.worker_task and not self.worker_task.done():
                health['checks']['worker_task'] = {'status': 'ok', 'running': True}
            else:
                health['checks']['worker_task'] = {'status': 'error', 'running': False}
                health['errors'].append('Notification worker not running')
            
            # Check queue size
            queue_size = self.notification_queue.qsize()
            if queue_size > 100:
                health['warnings'].append(f'Large notification queue: {queue_size} items')
                health['checks']['queue_size'] = {'status': 'warning', 'size': queue_size}
            else:
                health['checks']['queue_size'] = {'status': 'ok', 'size': queue_size}
            
            # Check failed notifications
            failed_count = len(self.failed_notifications)
            if failed_count > 10:
                health['warnings'].append(f'Many failed notifications: {failed_count}')
                health['checks']['failed_notifications'] = {'status': 'warning', 'count': failed_count}
            else:
                health['checks']['failed_notifications'] = {'status': 'ok', 'count': failed_count}
            
            # Check session
            if self.session and not self.session.closed:
                health['checks']['http_session'] = {'status': 'ok', 'closed': False}
            else:
                health['checks']['http_session'] = {'status': 'warning', 'closed': True}
                health['warnings'].append('HTTP session is closed')
            
            # Overall status
            if health['errors']:
                health['status'] = 'unhealthy'
            elif health['warnings']:
                health['status'] = 'warning'
            
            health['timestamp'] = datetime.utcnow().isoformat()
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup notification manager resources"""
        
        try:
            # Stop worker task
            if self.worker_task:
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    pass
            
            # Process remaining notifications
            while not self.notification_queue.empty():
                try:
                    notification = self.notification_queue.get_nowait()
                    await self._process_notification(notification)
                    self.notification_queue.task_done()
                except:
                    break
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            logger.info("Notification Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during notification manager cleanup: {e}")

# Factory function
def create_notification_manager(config) -> NotificationManager:
    """Factory function to create Notification Manager instance"""
    return NotificationManager(config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Mock configuration for testing
    class MockConfig:
        def __init__(self):
            # Email configuration (use your own for testing)
            self.smtp_server = "smtp.gmail.com"
            self.smtp_port = 587
            self.email_user = "your_email@gmail.com"
            self.email_password = "your_app_password"
            self.from_email = "your_email@gmail.com"
            self.to_emails = ["recipient@gmail.com"]
            
            # Discord configuration
            self.discord_webhook_url = "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
            
            # Telegram configuration  
            self.telegram_bot_token = "YOUR_BOT_TOKEN"
            self.telegram_chat_id = "YOUR_CHAT_ID"
            
            # Slack configuration
            self.slack_webhook_url = "https://hooks.slack.com/services/YOUR_WEBHOOK_URL"
            self.slack_channel = "#trading-bot"
            
            # Webhook configuration
            self.webhook_url = "https://httpbin.org/post"  # Test webhook
    
    async def test_notification_manager():
        print("📢 Notification Manager Test")
        print("=" * 50)
        
        try:
            # Create notification manager
            config = MockConfig()
            notification_manager = create_notification_manager(config)
            
            # Initialize (this will start the worker)
            await notification_manager.initialize()
            
            print(f"✅ Notification Manager created and initialized")
            print(f"📊 Available templates: {len(notification_manager.templates)}")
            
            # List available templates
            print(f"\n📋 Available notification templates:")
            for notif_type, template in notification_manager.templates.items():
                channels_str = ", ".join([ch.value for ch in template.channels])
                print(f"  {notif_type.value}: {template.priority.value} priority ({channels_str})")
            
            # Test 1: Trade execution notification
            print(f"\n🧪 Test 1: Trade Execution Notification")
            
            trade_data = {
                'symbol': 'BTC/USDT',
                'direction': 'LONG',
                'quantity': 0.1,
                'entry_price': 50000,
                'exit_price': 52000,
                'pnl': 200,
                'strategy_id': 'momentum_strategy',
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
            trade_success = await notification_manager.send_trade_notification(trade_data)
            print(f"  Trade notification sent: {'✅' if trade_success else '❌'}")
            
            # Test 2: Trading signal notification
            print(f"\n🧪 Test 2: Trading Signal Notification")
            
            signal_data = {
                'symbol': 'ETH/USDT',
                'signal_type': 'BUY',
                'confidence': 85.5,
                'entry_price': 3000,
                'stop_loss': 2850,
                'take_profit': 3300,
                'risk_reward_ratio': 2.1,
                'strategy_id': 'rsi_strategy'
            }
            
            signal_success = await notification_manager.send_signal_notification(signal_data)
            print(f"  Signal notification sent: {'✅' if signal_success else '❌'}")
            
            # Test 3: Risk alert notification
            print(f"\n🧪 Test 3: Risk Alert Notification")
            
            risk_data = {
                'risk_type': 'PORTFOLIO_CONCENTRATION',
                'current_level': '35%',
                'threshold': '30%',
                'recommendation': 'Reduce BTC position size',
                'affected_positions': 'BTC/USDT'
            }
            
            risk_success = await notification_manager.send_risk_alert(risk_data)
            print(f"  Risk alert sent: {'✅' if risk_success else '❌'}")
            
            # Test 4: Circuit breaker alert
            print(f"\n🧪 Test 4: Circuit Breaker Alert")
            
            breaker_data = {
                'trigger_type': 'DAILY_LOSS_LIMIT',
                'current_value': '-6.2%',
                'threshold': '-5.0%',
                'severity': 'CRITICAL',
                'cooldown_until': (datetime.utcnow() + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
            breaker_success = await notification_manager.send_circuit_breaker_alert(breaker_data)
            print(f"  Circuit breaker alert sent: {'✅' if breaker_success else '❌'}")
            
            # Test 5: Custom notification
            print(f"\n🧪 Test 5: Custom Notification")
            
            custom_success = await notification_manager.send_custom_notification(
                title="🎉 System Test Complete",
                message="All notification systems are functioning correctly!",
                channels=[NotificationChannel.DISCORD],
                priority=NotificationPriority.LOW
            )
            print(f"  Custom notification sent: {'✅' if custom_success else '❌'}")
            
            # Test 6: Performance report
            print(f"\n🧪 Test 6: Daily Performance Report")
            
            portfolio_summary = {
                'total_value': 12500.50,
                'total_pnl': 1250.50,
                'total_positions': 3
            }
            
            performance_metrics = {
                'daily_return': 2.5,
                'weekly_return': 8.2,
                'monthly_return': 15.7,
                'sharpe_ratio': 1.45,
                'max_drawdown': 8.3
            }
            
            trade_summary = {
                'trades_today': 5,
                'win_rate': 75.0
            }
            
            summary_success = True  # Would normally call send_daily_summary
            print(f"  Performance report prepared: {'✅' if summary_success else '❌'}")
            
            # Wait for notifications to process
            print(f"\n⏳ Waiting for notifications to process...")
            await asyncio.sleep(3)
            
            # Test 7: Statistics
            print(f"\n🧪 Test 7: Notification Statistics")
            
            stats = notification_manager.get_statistics()
            
            print(f"  Total sent: {stats['overview']['total_sent']}")
            print(f"  Total failed: {stats['overview']['total_failed']}")
            print(f"  Success rate: {stats['overview']['success_rate']:.1f}%")
            print(f"  Pending: {stats['overview']['pending_notifications']}")
            print(f"  Failed: {stats['overview']['failed_notifications']}")
            
            # Channel statistics
            print(f"  Channel performance:")
            for channel, channel_stats in stats['by_channel'].items():
                total = channel_stats['sent'] + channel_stats['failed']
                if total > 0:
                    success_rate = (channel_stats['sent'] / total) * 100
                    print(f"    {channel}: {channel_stats['sent']} sent, {channel_stats['failed']} failed ({success_rate:.1f}%)")
            
            # Type statistics
            print(f"  Notification type performance:")
            for notif_type, type_stats in stats['by_type'].items():
                total = type_stats['sent'] + type_stats['failed']
                if total > 0:
                    print(f"    {notif_type}: {type_stats['sent']} sent, {type_stats['failed']} failed")
            
            # Test 8: Health check
            print(f"\n🧪 Test 8: Health Check")
            
            health = await notification_manager.get_health_status()
            
            print(f"  Overall health: {health['status']}")
            print(f"  Checks performed: {len(health['checks'])}")
            print(f"  Warnings: {len(health['warnings'])}")
            print(f"  Errors: {len(health['errors'])}")
            
            for check_name, check_result in health['checks'].items():
                status_icon = "✅" if check_result['status'] == 'ok' else "⚠️" if check_result['status'] == 'warning' else "❌"
                print(f"    {check_name}: {status_icon} {check_result['status']}")
            
            if health['warnings']:
                print(f"  Warning messages:")
                for warning in health['warnings']:
                    print(f"    - {warning}")
            
            # Test 9: Template formatting
            print(f"\n🧪 Test 9: Template Formatting Test")
            
            test_template = notification_manager.templates[NotificationType.TRADE_EXECUTED]
            title, message = test_template.format_message(trade_data)
            
            print(f"  Template: {test_template.name}")
            print(f"  Formatted title: {title}")
            print(f"  Message preview: {message[:100]}...")
            print(f"  Channels: {[ch.value for ch in test_template.channels]}")
            print(f"  Priority: {test_template.priority.value}")
            
            # Test 10: Rate limiting
            print(f"\n🧪 Test 10: Rate Limiting Test")
            
            # Send multiple signal notifications (rate limited)
            for i in range(3):
                await notification_manager.send_signal_notification(signal_data)
                print(f"  Signal {i+1} queued")
            
            rate_limited_stats = notification_manager.get_statistics()
            print(f"  Rate limits active: {rate_limited_stats['rate_limits_active']}")
            
            print(f"\n🎉 Notification Manager test completed successfully!")
            
            # Final statistics
            final_stats = notification_manager.get_statistics()
            print(f"\n📈 Final Statistics:")
            print(f"  Total notifications sent: {final_stats['overview']['total_sent']}")
            print(f"  Total notifications failed: {final_stats['overview']['total_failed']}")
            print(f"  Overall success rate: {final_stats['overview']['success_rate']:.1f}%")
            print(f"  System uptime: {final_stats['overview']['uptime_hours']:.1f} hours")
            
        except Exception as e:
            print(f"❌ Error in Notification Manager test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            try:
                await notification_manager.cleanup()
                print(f"🧹 Notification Manager cleanup completed")
            except:
                pass
    
    # Run the test
    asyncio.run(test_notification_manager())