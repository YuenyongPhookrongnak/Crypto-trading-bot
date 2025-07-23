from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

Base = declarative_base()

class TradeDirection(enum.Enum):
    """Trade Direction Enum"""
    LONG = "LONG"
    SHORT = "SHORT"

class TradeStatus(enum.Enum):
    """Trade Status Enum"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"

class OrderType(enum.Enum):
    """Order Type Enum"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"

class LogLevel(enum.Enum):
    """Log Level Enum"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MarketData(Base):
    """ตาราง market data สำหรับเก็บข้อมูล OHLCV"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    
    # OHLCV data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Additional market data
    quote_volume = Column(Float)  # Volume in quote currency
    trade_count = Column(Integer)  # Number of trades
    taker_buy_volume = Column(Float)  # Taker buy volume
    taker_buy_quote_volume = Column(Float)  # Taker buy quote volume
    
    # Technical indicators (computed)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_timestamp_desc', 'timestamp'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}', close={self.close_price})>"

class Trade(Base):
    """ตาราง trades สำหรับเก็บข้อมูลการเทรด"""
    __tablename__ = 'trades'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False, index=True)
    strategy_id = Column(String(50), ForeignKey('strategies.id'), index=True)
    direction = Column(Enum(TradeDirection), nullable=False)
    
    # Entry information
    entry_time = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Float, nullable=False)
    entry_order_id = Column(String(50))
    entry_order_type = Column(Enum(OrderType), default=OrderType.MARKET)
    quantity = Column(Float, nullable=False)
    
    # Exit information
    exit_time = Column(DateTime, index=True)
    exit_price = Column(Float)
    exit_order_id = Column(String(50))
    exit_order_type = Column(Enum(OrderType))
    exit_reason = Column(String(50))  # TAKE_PROFIT, STOP_LOSS, MANUAL, TIME_LIMIT
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    trailing_stop = Column(Float)
    initial_stop_loss = Column(Float)  # Original stop loss
    initial_take_profit = Column(Float)  # Original take profit
    
    # Performance metrics
    pnl = Column(Float)  # Profit/Loss in quote currency
    pnl_percentage = Column(Float)  # P&L as percentage of capital
    commission = Column(Float)  # Total commission paid
    entry_commission = Column(Float)  # Entry commission
    exit_commission = Column(Float)  # Exit commission
    
    # Position details
    leverage = Column(Float, default=1.0)  # For futures trading
    margin_used = Column(Float)  # Margin used for position
    unrealized_pnl = Column(Float)  # Current unrealized P&L (for open positions)
    
    # Trade metadata
    status = Column(Enum(TradeStatus), default=TradeStatus.OPEN, index=True)
    market_condition = Column(String(20))  # TRENDING, RANGING, VOLATILE
    confidence_score = Column(Integer)  # Strategy confidence (0-100)
    ai_analysis = Column(JSON)  # AI analysis results
    strategy_signals = Column(JSON)  # Strategy signals that triggered trade
    
    # Risk metrics
    risk_amount = Column(Float)  # Amount at risk
    risk_percentage = Column(Float)  # Risk as percentage of portfolio
    position_size_percentage = Column(Float)  # Position size as percentage of portfolio
    correlation_risk = Column(Float)  # Correlation with other positions
    
    # Execution details
    slippage = Column(Float)  # Execution slippage
    execution_time_ms = Column(Integer)  # Time taken to execute order
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="trades")
    orders = relationship("Order", back_populates="trade", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_symbol_status', 'symbol', 'status'),
        Index('idx_entry_time_desc', 'entry_time'),
        Index('idx_strategy_status', 'strategy_id', 'status'),
        Index('idx_exit_time', 'exit_time'),
        Index('idx_pnl', 'pnl'),
    )
    
    @property
    def holding_period(self):
        """คำนวณระยะเวลา holding (ในชั่วโมง)"""
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600
        elif self.entry_time:
            return (datetime.utcnow() - self.entry_time).total_seconds() / 3600
        return 0
    
    @property
    def risk_reward_ratio(self):
        """คำนวณ Risk/Reward ratio"""
        if self.stop_loss and self.take_profit:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            return reward / risk if risk > 0 else None
        return None
    
    def __repr__(self):
        return f"<Trade(id='{self.id[:8]}...', symbol='{self.symbol}', direction='{self.direction.value}', status='{self.status.value}')>"

class Strategy(Base):
    """ตาราง strategies สำหรับเก็บข้อมูลกลยุทธ์การเทรด"""
    __tablename__ = 'strategies'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50))  # TREND_FOLLOWING, MEAN_REVERSION, MOMENTUM, etc.
    version = Column(String(20), default='1.0')
    parameters = Column(JSON)
    enabled = Column(Boolean, default=True, index=True)
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0)
    gross_profit = Column(Float, default=0)
    gross_loss = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    max_consecutive_wins = Column(Integer, default=0)
    max_consecutive_losses = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)  # Current win/loss streak
    
    # Advanced metrics
    sharpe_ratio = Column(Float, default=0)
    sortino_ratio = Column(Float, default=0)
    calmar_ratio = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    kelly_criterion = Column(Float, default=0)
    
    # Risk metrics
    var_95 = Column(Float, default=0)  # Value at Risk 95%
    expected_shortfall = Column(Float, default=0)
    beta = Column(Float, default=0)  # Beta vs market
    
    # Time-based performance
    avg_trade_duration = Column(Float, default=0)  # Average holding period in hours
    best_trade = Column(Float, default=0)  # Best single trade P&L
    worst_trade = Column(Float, default=0)  # Worst single trade P&L
    
    # Portfolio allocation
    allocation_percentage = Column(Float, default=0)  # Percentage of portfolio allocated
    max_position_size = Column(Float, default=0.1)  # Max position size as % of portfolio
    
    # Strategy state
    last_signal_time = Column(DateTime)
    last_trade_time = Column(DateTime)
    signals_generated = Column(Integer, default=0)
    signals_executed = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trades = relationship("Trade", back_populates="strategy", cascade="all, delete-orphan")
    
    @property
    def win_rate(self):
        """คำนวณ win rate"""
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
    
    @property
    def average_win(self):
        """คำนวณกำไรเฉลี่ยต่อการชนะ"""
        return self.gross_profit / self.winning_trades if self.winning_trades > 0 else 0
    
    @property
    def average_loss(self):
        """คำนวณขาดทุนเฉลี่ยต่อการแพ้"""
        return abs(self.gross_loss) / self.losing_trades if self.losing_trades > 0 else 0
    
    def __repr__(self):
        return f"<Strategy(id='{self.id}', name='{self.name}', enabled={self.enabled})>"

class Order(Base):
    """ตาราง orders สำหรับเก็บข้อมูลคำสั่งซื้อขาย"""
    __tablename__ = 'orders'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    trade_id = Column(String(36), ForeignKey('trades.id'), nullable=False, index=True)
    exchange_order_id = Column(String(50), unique=True, index=True)  # Order ID from exchange
    
    # Order details
    symbol = Column(String(20), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    side = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float)  # For limit orders
    stop_price = Column(Float)  # For stop orders
    
    # Execution details
    status = Column(String(20), default='PENDING')  # PENDING, FILLED, CANCELLED, REJECTED
    filled_quantity = Column(Float, default=0)
    remaining_quantity = Column(Float)
    average_price = Column(Float)
    
    # Fees and costs
    commission = Column(Float, default=0)
    commission_asset = Column(String(10))
    
    # Timestamps
    order_time = Column(DateTime, default=datetime.utcnow)
    execution_time = Column(DateTime)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional metadata
    time_in_force = Column(String(10), default='GTC')  # GTC, IOC, FOK
    client_order_id = Column(String(50))
    
    # Relationship
    trade = relationship("Trade", back_populates="orders")
    
    __table_args__ = (
        Index('idx_trade_order_type', 'trade_id', 'order_type'),
        Index('idx_status_symbol', 'status', 'symbol'),
        Index('idx_order_time', 'order_time'),
    )
    
    def __repr__(self):
        return f"<Order(id='{self.id[:8]}...', symbol='{self.symbol}', type='{self.order_type.value}', status='{self.status}')>"

class AIAnalysisLog(Base):
    """ตาราง AI analysis logs สำหรับเก็บผลการวิเคราะห์ของ AI"""
    __tablename__ = 'ai_analysis_log'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)  # TRADING_OPPORTUNITY, RISK_ASSESSMENT, MARKET_ANALYSIS
    
    # Input and output
    input_data = Column(JSON)  # Input data sent to AI
    ai_response = Column(JSON)  # Complete AI response
    processed_response = Column(JSON)  # Processed/cleaned response
    
    # Analysis metadata
    confidence_score = Column(Integer)  # AI confidence (0-100)
    market_condition = Column(String(20))  # Market condition at time of analysis
    processing_time = Column(Float)  # Processing time in seconds
    prompt_tokens = Column(Integer)  # Number of tokens in prompt
    completion_tokens = Column(Integer)  # Number of tokens in completion
    total_tokens = Column(Integer)  # Total tokens used
    
    # Model information
    model_name = Column(String(50))  # AI model used
    model_version = Column(String(20))  # Model version
    
    # Validation and feedback
    actual_outcome = Column(String(20))  # SUCCESS, FAILURE, PENDING
    accuracy_score = Column(Float)  # Accuracy of prediction (0-1)
    feedback_notes = Column(Text)  # Human feedback on analysis quality
    
    # Performance tracking
    prediction_horizon = Column(Integer)  # Prediction timeframe in minutes
    actual_price_move = Column(Float)  # Actual price movement observed
    predicted_direction = Column(String(10))  # UP, DOWN, SIDEWAYS
    actual_direction = Column(String(10))  # Actual direction observed
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_analysis_type_timestamp', 'analysis_type', 'timestamp'),
        Index('idx_confidence_score', 'confidence_score'),
        Index('idx_actual_outcome', 'actual_outcome'),
    )
    
    def __repr__(self):
        return f"<AIAnalysisLog(symbol='{self.symbol}', type='{self.analysis_type}', confidence={self.confidence_score})>"

class PerformanceMetric(Base):
    """ตาราง performance metrics สำหรับเก็บ metrics ประสิทธิภาพ"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String(50), nullable=False, index=True)  # DAILY, WEEKLY, MONTHLY, YEARLY
    date = Column(DateTime, nullable=False, index=True)
    
    # Core performance metrics
    total_return = Column(Float)
    annualized_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    max_drawdown = Column(Float)
    current_drawdown = Column(Float)
    
    # Trading metrics
    win_rate = Column(Float)
    profit_factor = Column(Float)
    kelly_criterion = Column(Float)
    expectancy = Column(Float)  # Expected value per trade
    
    # Volume metrics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    avg_trade_duration = Column(Float)  # in hours
    
    # Risk metrics
    var_95 = Column(Float)  # Value at Risk 95%
    cvar_95 = Column(Float)  # Conditional Value at Risk 95%
    beta = Column(Float)  # Beta vs benchmark
    alpha = Column(Float)  # Alpha vs benchmark
    information_ratio = Column(Float)
    treynor_ratio = Column(Float)
    
    # Portfolio metrics
    portfolio_value = Column(Float)
    cash_balance = Column(Float)
    invested_amount = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    
    # Strategy allocation
    strategy_allocations = Column(JSON)  # Allocation per strategy
    
    # Market correlation
    market_correlation = Column(Float)  # Correlation with market index
    
    # Detailed metrics in JSON format
    metrics_json = Column(JSON)  # Additional detailed metrics
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_metric_type_date', 'metric_type', 'date'),
        Index('idx_date_desc', 'date'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetric(type='{self.metric_type}', date='{self.date}', return={self.total_return})>"

class SystemLog(Base):
    """ตาราง system logs สำหรับเก็บ log ของระบบ"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    level = Column(Enum(LogLevel), nullable=False, index=True)
    component = Column(String(50), nullable=False, index=True)  # SCANNER, STRATEGY, RISK_MANAGER, etc.
    message = Column(Text, nullable=False)
    details = Column(JSON)  # Additional details in JSON format
    
    # Context information
    symbol = Column(String(20), index=True)  # Related symbol if applicable
    strategy_id = Column(String(50))  # Related strategy if applicable
    trade_id = Column(String(36))  # Related trade if applicable
    
    # Performance impact
    execution_time_ms = Column(Integer)  # Execution time in milliseconds
    memory_usage_mb = Column(Float)  # Memory usage in MB
    cpu_usage_percent = Column(Float)  # CPU usage percentage
    
    # Error details (for ERROR and CRITICAL levels)
    error_code = Column(String(20))
    stack_trace = Column(Text)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_level_timestamp', 'level', 'timestamp'),
        Index('idx_component_timestamp', 'component', 'timestamp'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timestamp_desc', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemLog(level='{self.level.value}', component='{self.component}', timestamp='{self.timestamp}')>"

class MarketCondition(Base):
    """ตาราง market conditions สำหรับเก็บสภาวะตลาด"""
    __tablename__ = 'market_conditions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Overall market sentiment
    market_sentiment = Column(String(20))  # BULLISH, BEARISH, NEUTRAL
    volatility_regime = Column(String(20))  # LOW, MEDIUM, HIGH, EXTREME
    
    # Market indicators
    fear_greed_index = Column(Integer)  # 0-100
    vix_equivalent = Column(Float)  # Volatility index
    
    # Crypto-specific metrics
    btc_dominance = Column(Float)  # Bitcoin dominance percentage
    total_market_cap = Column(Float)  # Total crypto market cap
    defi_tvl = Column(Float)  # DeFi Total Value Locked
    
    # Technical indicators for overall market
    market_trend = Column(String(20))  # UPTREND, DOWNTREND, SIDEWAYS
    trend_strength = Column(Float)  # 0-1
    support_level = Column(Float)  # Market support level
    resistance_level = Column(Float)  # Market resistance level
    
    # Volume analysis
    total_volume_24h = Column(Float)
    volume_trend = Column(String(20))  # INCREASING, DECREASING, STABLE
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_timestamp_market_sentiment', 'timestamp', 'market_sentiment'),
        Index('idx_volatility_regime', 'volatility_regime'),
    )
    
    def __repr__(self):
        return f"<MarketCondition(timestamp='{self.timestamp}', sentiment='{self.market_sentiment}', volatility='{self.volatility_regime}')>"

# Helper functions for database operations
def create_all_tables(engine):
    """สร้างตารางทั้งหมด"""
    Base.metadata.create_all(bind=engine)

def drop_all_tables(engine):
    """ลบตารางทั้งหมด"""
    Base.metadata.drop_all(bind=engine)

if __name__ == "__main__":
    # ทดสอบ models
    print("🗄️  Database Models Test")
    print("=" * 50)
    
    # แสดงข้อมูล tables
    tables = Base.metadata.tables.keys()
    print(f"📋 Tables defined: {len(tables)}")
    for table in sorted(tables):
        print(f"  • {table}")
    
    print("\n🔍 Model relationships:")
    print("  • Strategy -> Trade (1:many)")
    print("  • Trade -> Order (1:many)")
    print("  • All models include proper indexes")
    print("  • Enums for type safety")
    print("  • JSON fields for flexible data")
    
    print("\n✅ Models definition complete!")