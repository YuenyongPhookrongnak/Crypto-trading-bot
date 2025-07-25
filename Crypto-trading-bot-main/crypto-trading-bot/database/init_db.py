#!/usr/bin/env python3
"""
Database Initialization and Setup Script
Handles database creation, table setup, initial data seeding, and migrations
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import DatabaseManager, initialize_database_manager
from database.models import (
    Base, Strategy, MarketCondition, PerformanceMetric, 
    TradeDirection, TradeStatus, OrderType, LogLevel,
    create_all_tables, drop_all_tables
)
from config.settings import config

logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Database Initialization and Setup Manager"""
    
    def __init__(self, database_url: str = None, force_recreate: bool = False):
        self.database_url = database_url or config.system.database_url
        self.force_recreate = force_recreate
        self.db_manager = None
        
    def initialize(self) -> bool:
        """Initialize complete database setup"""
        
        try:
            print("🗄️  Crypto Trading Bot - Database Initialization")
            print("=" * 60)
            
            # Step 1: Initialize database manager
            print("📡 Initializing database connection...")
            self.db_manager = initialize_database_manager(self.database_url)
            
            # Step 2: Health check
            health = self.db_manager.health_check()
            if health['status'] != 'healthy':
                print(f"❌ Database health check failed: {health.get('error', 'Unknown error')}")
                return False
            
            print(f"✅ Database connection established ({health['database_type']})")
            
            # Step 3: Create/recreate tables
            if self.force_recreate:
                print("🔄 Dropping existing tables...")
                self._drop_tables()
            
            print("🏗️  Creating database tables...")
            success = self._create_tables()
            if not success:
                return False
            
            # Step 4: Seed initial data
            print("🌱 Seeding initial data...")
            self._seed_initial_data()
            
            # Step 5: Create indexes and optimize
            print("⚡ Creating indexes and optimizing...")
            self._create_indexes()
            self._optimize_database()
            
            # Step 6: Validate setup
            print("✅ Validating database setup...")
            validation_result = self._validate_setup()
            
            if validation_result['valid']:
                print("\n🎉 Database initialization completed successfully!")
                self._print_setup_summary(validation_result)
                return True
            else:
                print(f"\n❌ Database validation failed: {validation_result['errors']}")
                return False
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            print(f"\n💥 Database initialization failed: {e}")
            return False
        
        finally:
            if self.db_manager:
                self.db_manager.close()
    
    def _create_tables(self) -> bool:
        """Create all database tables"""
        
        try:
            create_all_tables(self.db_manager.engine)
            
            # Verify tables were created
            inspector = self.db_manager.engine.dialect.get_table_names(
                self.db_manager.engine.connect()
            )
            
            expected_tables = [
                'market_data', 'trades', 'strategies', 'orders',
                'ai_analysis_log', 'performance_metrics', 
                'system_logs', 'market_conditions'
            ]
            
            missing_tables = [table for table in expected_tables if table not in inspector]
            
            if missing_tables:
                print(f"❌ Missing tables: {missing_tables}")
                return False
            
            print(f"✅ Created {len(expected_tables)} tables successfully")
            return True
            
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            print(f"❌ Table creation failed: {e}")
            return False
    
    def _drop_tables(self):
        """Drop all existing tables"""
        
        try:
            drop_all_tables(self.db_manager.engine)
            print("✅ Existing tables dropped")
            
        except Exception as e:
            logger.warning(f"Error dropping tables (may not exist): {e}")
    
    def _seed_initial_data(self):
        """Seed database with initial data"""
        
        try:
            with self.db_manager.get_session() as session:
                
                # Seed default strategies
                self._seed_strategies(session)
                
                # Seed initial market conditions
                self._seed_market_conditions(session)
                
                # Seed initial performance metrics
                self._seed_performance_metrics(session)
                
                print("✅ Initial data seeded successfully")
                
        except Exception as e:
            logger.error(f"Data seeding failed: {e}")
            print(f"❌ Data seeding failed: {e}")
    
    def _seed_strategies(self, session):
        """Seed default trading strategies"""
        
        strategies_data = [
            {
                'id': 'rsi_strategy',
                'name': 'RSI Mean Reversion Strategy',
                'description': 'Trading strategy based on RSI overbought/oversold conditions with trend confirmation',
                'category': 'MEAN_REVERSION',
                'version': '1.0',
                'parameters': config.strategies['rsi_strategy'],
                'enabled': config.strategies['rsi_strategy'].get('enabled', True),
                'allocation_percentage': 0.33,
                'max_position_size': 0.1
            },
            {
                'id': 'volume_profile_strategy', 
                'name': 'Volume Profile Breakout Strategy',
                'description': 'Strategy that trades breakouts from volume profile key levels with volume confirmation',
                'category': 'BREAKOUT',
                'version': '1.0',
                'parameters': config.strategies['volume_profile_strategy'],
                'enabled': config.strategies['volume_profile_strategy'].get('enabled', True),
                'allocation_percentage': 0.33,
                'max_position_size': 0.1
            },
            {
                'id': 'multi_timeframe_strategy',
                'name': 'Multi-Timeframe Trend Strategy',
                'description': 'Advanced strategy using multiple timeframes for trend identification and entry timing',
                'category': 'TREND_FOLLOWING',
                'version': '1.0',
                'parameters': config.strategies['multi_timeframe_strategy'],
                'enabled': config.strategies['multi_timeframe_strategy'].get('enabled', True),
                'allocation_percentage': 0.34,
                'max_position_size': 0.15
            }
        ]
        
        for strategy_data in strategies_data:
            # Check if strategy already exists
            existing = session.query(Strategy).filter(
                Strategy.id == strategy_data['id']
            ).first()
            
            if not existing:
                strategy = Strategy(**strategy_data)
                session.add(strategy)
                print(f"  • Created strategy: {strategy_data['name']}")
            else:
                print(f"  • Strategy already exists: {strategy_data['name']}")
    
    def _seed_market_conditions(self, session):
        """Seed initial market conditions"""
        
        # Check if market conditions already exist
        existing_conditions = session.query(MarketCondition).first()
        
        if not existing_conditions:
            initial_condition = MarketCondition(
                timestamp=datetime.utcnow(),
                market_sentiment='NEUTRAL',
                volatility_regime='MEDIUM',
                fear_greed_index=50,
                btc_dominance=45.0,
                total_market_cap=1000000000000,  # 1T USD
                market_trend='SIDEWAYS',
                trend_strength=0.5,
                total_volume_24h=50000000000,  # 50B USD
                volume_trend='STABLE'
            )
            
            session.add(initial_condition)
            print("  • Created initial market conditions")
        else:
            print("  • Market conditions already exist")
    
    def _seed_performance_metrics(self, session):
        """Seed initial performance metrics"""
        
        # Check if performance metrics already exist
        existing_metrics = session.query(PerformanceMetric).first()
        
        if not existing_metrics:
            initial_metrics = PerformanceMetric(
                metric_type='DAILY',
                date=datetime.utcnow().date(),
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                portfolio_value=config.trading.initial_capital,
                cash_balance=config.trading.initial_capital,
                invested_amount=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                metrics_json={'initialization': True}
            )
            
            session.add(initial_metrics)
            print(f"  • Created initial performance metrics (Capital: ${config.trading.initial_capital:,.2f})")
        else:
            print("  • Performance metrics already exist")
    
    def _create_indexes(self):
        """Create additional database indexes for performance"""
        
        try:
            additional_indexes = [
                # Market data indexes
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe_timestamp ON market_data(symbol, timeframe, timestamp DESC);",
                
                # Trades indexes
                "CREATE INDEX IF NOT EXISTS idx_trades_entry_time_desc ON trades(entry_time DESC);",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_status_entry_time ON trades(symbol, status, entry_time);",
                "CREATE INDEX IF NOT EXISTS idx_trades_strategy_pnl ON trades(strategy_id, pnl);",
                
                # System logs indexes
                "CREATE INDEX IF NOT EXISTS idx_system_logs_level_timestamp ON system_logs(level, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_system_logs_component_timestamp ON system_logs(component, timestamp DESC);",
                
                # AI analysis logs indexes
                "CREATE INDEX IF NOT EXISTS idx_ai_analysis_confidence_timestamp ON ai_analysis_log(confidence_score, timestamp DESC);",
                
                # Performance metrics indexes
                "CREATE INDEX IF NOT EXISTS idx_performance_metrics_date_type ON performance_metrics(date DESC, metric_type);"
            ]
            
            for index_sql in additional_indexes:
                try:
                    self.db_manager.execute_query(index_sql)
                except Exception as e:
                    # Ignore "already exists" errors
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Index creation warning: {e}")
            
            print("✅ Additional indexes created")
            
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")
    
    def _optimize_database(self):
        """Optimize database for performance"""
        
        try:
            self.db_manager.optimize_database()
            print("✅ Database optimized")
            
        except Exception as e:
            logger.warning(f"Database optimization failed: {e}")
    
    def _validate_setup(self) -> Dict[str, Any]:
        """Validate database setup"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            with self.db_manager.get_session() as session:
                
                # Check tables exist and have data
                table_counts = {}
                
                # Check strategies
                strategy_count = session.query(Strategy).count()
                table_counts['strategies'] = strategy_count
                if strategy_count == 0:
                    validation_result['errors'].append("No strategies found in database")
                    validation_result['valid'] = False
                
                # Check if at least one strategy is enabled
                enabled_strategies = session.query(Strategy).filter(Strategy.enabled == True).count()
                table_counts['enabled_strategies'] = enabled_strategies
                if enabled_strategies == 0:
                    validation_result['errors'].append("No strategies are enabled")
                    validation_result['valid'] = False
                
                # Check performance metrics
                metrics_count = session.query(PerformanceMetric).count()
                table_counts['performance_metrics'] = metrics_count
                if metrics_count == 0:
                    validation_result['warnings'].append("No performance metrics found")
                
                # Check market conditions
                conditions_count = session.query(MarketCondition).count()
                table_counts['market_conditions'] = conditions_count
                if conditions_count == 0:
                    validation_result['warnings'].append("No market conditions found")
                
                validation_result['statistics'] = table_counts
                
                # Database connectivity test
                health = self.db_manager.health_check()
                if health['status'] != 'healthy':
                    validation_result['errors'].append(f"Database health check failed: {health.get('error')}")
                    validation_result['valid'] = False
                
        except Exception as e:
            validation_result['errors'].append(f"Validation query failed: {e}")
            validation_result['valid'] = False
        
        return validation_result
    
    def _print_setup_summary(self, validation_result: Dict[str, Any]):
        """Print setup summary"""
        
        print("\n📊 Database Setup Summary:")
        print("-" * 40)
        
        stats = validation_result.get('statistics', {})
        print(f"📈 Strategies: {stats.get('strategies', 0)} (Enabled: {stats.get('enabled_strategies', 0)})")
        print(f"📊 Performance Metrics: {stats.get('performance_metrics', 0)}")
        print(f"🌍 Market Conditions: {stats.get('market_conditions', 0)}")
        
        if validation_result.get('warnings'):
            print(f"\n⚠️  Warnings:")
            for warning in validation_result['warnings']:
                print(f"  • {warning}")
        
        # Database info
        health = self.db_manager.health_check()
        print(f"\n🗄️  Database Info:")
        print(f"  • Type: {health.get('database_type', 'Unknown')}")
        print(f"  • URL: {self.db_manager._mask_credentials(self.database_url)}")
        print(f"  • Response Time: {health.get('response_time_ms', 'N/A')} ms")
        
        print(f"\n💰 Initial Configuration:")
        print(f"  • Initial Capital: ${config.trading.initial_capital:,.2f}")
        print(f"  • Max Risk per Trade: {config.trading.max_risk_per_trade:.1%}")
        print(f"  • Max Open Positions: {config.trading.max_open_positions}")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Configure your API keys in .env file")
        print(f"  2. Test strategies with paper trading: python run_bot.py --paper-trading")
        print(f"  3. Monitor performance and adjust parameters")
        print(f"  4. Enable live trading when ready")

def migrate_database(database_url: str = None):
    """Perform database migration (placeholder for future migrations)"""
    
    print("🔄 Database Migration")
    print("-" * 30)
    
    try:
        db_manager = initialize_database_manager(database_url)
        
        # Future migrations will be implemented here
        # For now, just validate current schema
        
        health = db_manager.health_check()
        if health['status'] == 'healthy':
            print("✅ Database schema is up to date")
            return True
        else:
            print(f"❌ Migration failed: {health.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False

def backup_database(backup_path: str = None, database_url: str = None):
    """Create database backup"""
    
    backup_path = backup_path or f"backups/trading_bot_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"💾 Creating Database Backup")
    print("-" * 40)
    print(f"Backup path: {backup_path}")
    
    try:
        db_manager = initialize_database_manager(database_url)
        
        success = db_manager.backup_database(backup_path)
        
        if success:
            print("✅ Database backup created successfully")
            return True
        else:
            print("❌ Database backup failed")
            return False
            
    except Exception as e:
        print(f"❌ Backup error: {e}")
        return False

def reset_database(database_url: str = None, confirm: bool = False):
    """Reset database (drop and recreate all tables)"""
    
    if not confirm:
        response = input("⚠️  This will delete all data. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Database reset cancelled")
            return False
    
    print("🔄 Resetting Database")
    print("-" * 30)
    
    initializer = DatabaseInitializer(database_url, force_recreate=True)
    return initializer.initialize()

def main():
    """Main entry point for database initialization"""
    
    parser = argparse.ArgumentParser(description='Database Initialization and Management')
    parser.add_argument('--database-url', help='Database URL (overrides config)')
    parser.add_argument('--force-recreate', action='store_true', 
                       help='Drop and recreate all tables')
    parser.add_argument('--migrate', action='store_true',
                       help='Run database migration')
    parser.add_argument('--backup', help='Create database backup')
    parser.add_argument('--reset', action='store_true',
                       help='Reset database (drop all data)')
    parser.add_argument('--yes', action='store_true',
                       help='Auto-confirm dangerous operations')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.migrate:
            success = migrate_database(args.database_url)
            
        elif args.backup:
            success = backup_database(args.backup, args.database_url)
            
        elif args.reset:
            success = reset_database(args.database_url, args.yes)
            
        else:
            # Default: initialize database
            initializer = DatabaseInitializer(
                database_url=args.database_url,
                force_recreate=args.force_recreate
            )
            success = initializer.initialize()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()