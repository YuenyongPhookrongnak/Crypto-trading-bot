"""
Database Connection Management

This module handles database connections, session management, and provides
utilities for database operations throughout the trading bot system.
"""

import logging
import asyncio
from typing import Optional, AsyncGenerator, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
import os

try:
    from sqlalchemy import create_engine, MetaData, inspect
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import event
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    # Create dummy aiosqlite for fallback
    class DummyAioSQLite:
        @staticmethod
        def connect(database):
            class DummyConnection:
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                async def execute(self, query, params=None):
                    pass
                async def commit(self):
                    pass
            return DummyConnection()
    
    aiosqlite = DummyAioSQLite()

logger = logging.getLogger(__name__)

# Base class for models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
else:
    # Dummy base for when SQLAlchemy is not available
    class Base:
        pass

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str = None, echo: bool = False):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
        self.echo = echo
        
        # Database engines
        self.engine = None
        self.async_engine = None
        
        # Session factories
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        
        # Connection state
        self.is_initialized = False
        self.is_connected = False
        
        # Connection pool settings
        self.pool_size = 5
        self.max_overflow = 10
        self.pool_timeout = 30
        
        logger.info(f"Database manager initialized with URL: {self._mask_password(self.database_url)}")
    
    def _mask_password(self, url: str) -> str:
        """Mask password in database URL for logging"""
        try:
            if '@' in url and ':' in url:
                parts = url.split('@')
                if len(parts) == 2:
                    auth_part = parts[0]
                    if ':' in auth_part:
                        auth_parts = auth_part.split(':')
                        if len(auth_parts) >= 3:  # protocol:user:password
                            masked = ':'.join(auth_parts[:-1]) + ':***'
                            return masked + '@' + parts[1]
            return url
        except:
            return url
    
    async def initialize(self) -> bool:
        """Initialize database connections"""
        try:
            if not SQLALCHEMY_AVAILABLE:
                logger.warning("SQLAlchemy not available - using fallback database manager")
                return await self._initialize_fallback()
            
            # Create engines
            await self._create_engines()
            
            # Create session factories
            self._create_session_factories()
            
            # Test connections
            await self._test_connections()
            
            # Create tables if they don't exist
            await self._create_tables()
            
            self.is_initialized = True
            self.is_connected = True
            
            logger.info("Database manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            return False
    
    async def _initialize_fallback(self) -> bool:
        """Initialize fallback database manager without SQLAlchemy"""
        try:
            # Simple SQLite connection for fallback
            if 'sqlite' in self.database_url.lower():
                db_path = self.database_url.replace('sqlite:///', '').replace('sqlite:', '')
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
                
                # Test connection
                if AIOSQLITE_AVAILABLE:
                    async with aiosqlite.connect(db_path) as conn:
                        await conn.execute('SELECT 1')
                    logger.info("Fallback SQLite connection successful")
                else:
                    # Use Python's built-in sqlite3 as last resort
                    import sqlite3
                    try:
                        conn = sqlite3.connect(db_path)
                        conn.execute('SELECT 1')
                        conn.close()
                        logger.info("Basic SQLite connection successful (sync mode)")
                    except Exception as e:
                        logger.warning(f"SQLite connection test failed: {e}")
                        # Create empty file if it doesn't exist
                        if not os.path.exists(db_path):
                            open(db_path, 'a').close()
                        logger.info("Basic SQLite file created")
                
                self.is_initialized = True
                self.is_connected = True
                return True
            
            else:
                logger.error("Fallback mode only supports SQLite")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize fallback database: {e}")
            return False
    
    async def _create_engines(self):
        """Create database engines"""
        try:
            # Determine if we need async engine
            if 'sqlite' in self.database_url.lower():
                # SQLite configuration
                if self.database_url.startswith('sqlite:///'):
                    # Async SQLite
                    async_url = self.database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
                    self.async_engine = create_async_engine(
                        async_url,
                        echo=self.echo,
                        poolclass=StaticPool,
                        connect_args={
                            "check_same_thread": False,
                            "timeout": 20
                        }
                    )
                
                # Sync SQLite
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    poolclass=StaticPool,
                    connect_args={"check_same_thread": False}
                )
                
            else:
                # PostgreSQL or other databases
                # Sync engine
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_timeout=self.pool_timeout
                )
                
                # Async engine (convert URL for async)
                if 'postgresql://' in self.database_url:
                    async_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
                elif 'mysql://' in self.database_url:
                    async_url = self.database_url.replace('mysql://', 'mysql+aiomysql://')
                else:
                    async_url = self.database_url
                
                self.async_engine = create_async_engine(
                    async_url,
                    echo=self.echo,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_timeout=self.pool_timeout
                )
            
            logger.debug("Database engines created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database engines: {e}")
            raise
    
    def _create_session_factories(self):
        """Create session factories"""
        try:
            # Sync session factory
            if self.engine:
                self.SessionLocal = sessionmaker(
                    bind=self.engine,
                    autocommit=False,
                    autoflush=False
                )
            
            # Async session factory
            if self.async_engine:
                self.AsyncSessionLocal = async_sessionmaker(
                    bind=self.async_engine,
                    class_=AsyncSession,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False
                )
            
            logger.debug("Session factories created successfully")
            
        except Exception as e:
            logger.error(f"Error creating session factories: {e}")
            raise
    
    async def _test_connections(self):
        """Test database connections"""
        try:
            # Test sync connection
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute("SELECT 1")
                logger.debug("Sync database connection test passed")
            
            # Test async connection
            if self.async_engine:
                async with self.async_engine.connect() as conn:
                    await conn.execute("SELECT 1")
                logger.debug("Async database connection test passed")
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        try:
            if self.async_engine and hasattr(Base, 'metadata'):
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created/verified")
            elif self.engine and hasattr(Base, 'metadata'):
                Base.metadata.create_all(bind=self.engine)
                logger.info("Database tables created/verified (sync)")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        if not self.AsyncSessionLocal:
            raise RuntimeError("Async session factory not initialized")
        
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_session(self):
        """Get sync database session (context manager)"""
        if not self.SessionLocal:
            raise RuntimeError("Session factory not initialized")
        
        return self.SessionLocal()
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results"""
        try:
            if self.async_engine:
                async with self.async_engine.connect() as conn:
                    result = await conn.execute(query, params or {})
                    return [dict(row) for row in result.fetchall()]
            elif self.engine:
                with self.engine.connect() as conn:
                    result = conn.execute(query, params or {})
                    return [dict(row) for row in result.fetchall()]
            else:
                raise RuntimeError("No database engine available")
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    async def execute_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        """Execute a SQL command (INSERT, UPDATE, DELETE)"""
        try:
            if self.async_engine:
                async with self.async_engine.connect() as conn:
                    await conn.execute(command, params or {})
                    await conn.commit()
            elif self.engine:
                with self.engine.connect() as conn:
                    conn.execute(command, params or {})
                    conn.commit()
            else:
                raise RuntimeError("No database engine available")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            if self.engine:
                inspector = inspect(self.engine)
                columns = inspector.get_columns(table_name)
                indexes = inspector.get_indexes(table_name)
                
                return {
                    'exists': table_name in inspector.get_table_names(),
                    'columns': columns,
                    'indexes': indexes
                }
            else:
                return {'exists': False, 'columns': [], 'indexes': []}
                
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {'exists': False, 'columns': [], 'indexes': []}
    
    async def backup_database(self, backup_path: str) -> bool:
        """Create database backup (SQLite only)"""
        try:
            if 'sqlite' not in self.database_url.lower():
                logger.warning("Database backup only supported for SQLite")
                return False
            
            # Get source database path
            source_path = self.database_url.replace('sqlite:///', '').replace('sqlite:', '')
            
            # Copy database file
            import shutil
            shutil.copy2(source_path, backup_path)
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                'is_connected': self.is_connected,
                'database_url': self._mask_password(self.database_url),
                'engine_type': 'async' if self.async_engine else 'sync' if self.engine else 'none'
            }
            
            if self.engine:
                inspector = inspect(self.engine)
                stats.update({
                    'tables': inspector.get_table_names(),
                    'table_count': len(inspector.get_table_names())
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            health = {
                'status': 'healthy',
                'is_initialized': self.is_initialized,
                'is_connected': self.is_connected,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Test connection
            if self.async_engine:
                async with self.async_engine.connect() as conn:
                    await conn.execute("SELECT 1")
                health['async_connection'] = 'ok'
            
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute("SELECT 1")
                health['sync_connection'] = 'ok'
            
            return health
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup database connections"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                logger.debug("Async engine disposed")
            
            if self.engine:
                self.engine.dispose()
                logger.debug("Sync engine disposed")
            
            self.is_connected = False
            logger.info("Database manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
async def init_database(database_url: str = None, echo: bool = False) -> bool:
    """Initialize database"""
    global db_manager
    if database_url:
        db_manager = DatabaseManager(database_url, echo)
    return await db_manager.initialize()

async def get_db_session():
    """Get database session (async context manager)"""
    async with db_manager.get_async_session() as session:
        yield session

def get_sync_db_session():
    """Get synchronous database session"""
    return db_manager.get_session()

async def execute_query(query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Execute query"""
    return await db_manager.execute_query(query, params)

async def execute_command(command: str, params: Dict[str, Any] = None) -> bool:
    """Execute command"""
    return await db_manager.execute_command(command, params)

async def close_database():
    """Close database connections"""
    await db_manager.cleanup()

# Simple fallback database operations for when SQLAlchemy is not available
class SimpleDatabaseManager:
    """Simple database manager for when SQLAlchemy is not available"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize simple database"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
            
            if AIOSQLITE_AVAILABLE:
                async with aiosqlite.connect(self.db_path) as conn:
                    # Create basic tables
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS trades (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            direction TEXT NOT NULL,
                            quantity REAL NOT NULL,
                            entry_price REAL NOT NULL,
                            exit_price REAL,
                            pnl REAL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS system_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            level TEXT NOT NULL,
                            message TEXT NOT NULL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    await conn.commit()
                
                self.is_initialized = True
                logger.info(f"Simple database initialized: {self.db_path}")
                return True
            else:
                # Just create empty file
                if not os.path.exists(self.db_path):
                    open(self.db_path, 'a').close()
                self.is_initialized = True
                logger.info(f"Basic database file created: {self.db_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing simple database: {e}")
            return False
    
    async def log_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Log trade to database"""
        try:
            if not AIOSQLITE_AVAILABLE:
                return False
            
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('''
                    INSERT INTO trades (symbol, direction, quantity, entry_price, exit_price, pnl)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('symbol', ''),
                    trade_data.get('direction', ''),
                    trade_data.get('quantity', 0),
                    trade_data.get('entry_price', 0),
                    trade_data.get('exit_price'),
                    trade_data.get('pnl')
                ))
                await conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return False
    
    async def log_message(self, level: str, message: str) -> bool:
        """Log message to database"""
        try:
            if not AIOSQLITE_AVAILABLE:
                return False
            
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('''
                    INSERT INTO system_logs (level, message)
                    VALUES (?, ?)
                ''', (level, message))
                await conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging message: {e}")
            return False

# Create simple database manager as fallback
simple_db = SimpleDatabaseManager()

# Auto-initialize based on availability
async def auto_init_database() -> bool:
    """Auto-initialize database based on available libraries"""
    try:
        if SQLALCHEMY_AVAILABLE:
            return await init_database()
        else:
            return await simple_db.initialize()
    except Exception as e:
        logger.error(f"Auto database initialization failed: {e}")
        return False

# Example usage and testing
async def test_database_connection():
    """Test database connection and operations"""
    try:
        print("🗄️ Database Connection Test")
        print("=" * 40)
        
        # Initialize database
        success = await auto_init_database()
        print(f"Database initialization: {'✅ Success' if success else '❌ Failed'}")
        
        if SQLALCHEMY_AVAILABLE and success:
            # Get database stats
            stats = await db_manager.get_database_stats()
            print(f"Database type: {stats.get('engine_type', 'unknown')}")
            print(f"Tables: {stats.get('table_count', 0)}")
            
            # Health check
            health = await db_manager.health_check()
            print(f"Health status: {health['status']}")
            
            # Test query (if we have tables)
            try:
                result = await execute_query("SELECT 1 as test")
                print(f"Query test: ✅ Success ({len(result)} rows)")
            except Exception as e:
                print(f"Query test: ❌ Failed - {e}")
        
        elif success:
            print("Simple database mode active")
            
            # Test simple operations
            if AIOSQLITE_AVAILABLE:
                trade_data = {
                    'symbol': 'BTC/USDT',
                    'direction': 'BUY',
                    'quantity': 0.01,
                    'entry_price': 45000,
                    'pnl': 100
                }
                
                log_success = await simple_db.log_trade(trade_data)
                print(f"Trade logging test: {'✅ Success' if log_success else '❌ Failed'}")
        
        # Cleanup
        await close_database()
        print("✅ Database test completed")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_database_connection())