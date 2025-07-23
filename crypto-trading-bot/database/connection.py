import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from sqlalchemy import create_engine, text, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.pool import StaticPool
import time
from threading import Lock
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database Connection Manager with connection pooling and error handling"""
    
    def __init__(self, database_url: str = None, echo: bool = False):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')
        self.echo = echo
        self.engine = None
        self.SessionLocal = None
        self._lock = Lock()
        self._connection_pool_size = 5
        self._max_overflow = 10
        self._pool_timeout = 30
        self._pool_recycle = 3600  # 1 hour
        self._retry_attempts = 3
        self._retry_delay = 1  # seconds
        
        # Connection statistics
        self.stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'queries_executed': 0,
            'errors_count': 0,
            'last_error': None,
            'uptime_start': datetime.now()
        }
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine with proper configuration"""
        try:
            engine_kwargs = {
                'echo': self.echo,
                'future': True  # Use SQLAlchemy 2.0 style
            }
            
            # Configure based on database type
            if self.database_url.startswith('sqlite'):
                # SQLite specific configuration
                engine_kwargs.update({
                    'poolclass': StaticPool,
                    'connect_args': {
                        'check_same_thread': False,
                        'timeout': 20
                    }
                })
                
                # Ensure directory exists for SQLite
                db_path = self.database_url.replace('sqlite:///', '')
                os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
                
            elif self.database_url.startswith('postgresql'):
                # PostgreSQL specific configuration
                engine_kwargs.update({
                    'pool_size': self._connection_pool_size,
                    'max_overflow': self._max_overflow,
                    'pool_timeout': self._pool_timeout,
                    'pool_recycle': self._pool_recycle,
                    'pool_pre_ping': True,  # Verify connections before use
                    'connect_args': {
                        'connect_timeout': 10,
                        'application_name': 'CryptoTradingBot'
                    }
                })
            
            # Create engine
            self.engine = create_engine(self.database_url, **engine_kwargs)
            
            # Add event listeners for connection tracking
            self._setup_event_listeners()
            
            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
            
            logger.info(f"Database engine initialized: {self._mask_credentials(self.database_url)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Track new connections"""
            self.stats['connections_created'] += 1
            logger.debug("New database connection created")
            
            # SQLite specific optimizations
            if self.database_url.startswith('sqlite'):
                with dbapi_connection.cursor() as cursor:
                    # Enable WAL mode for better concurrency
                    cursor.execute("PRAGMA journal_mode=WAL")
                    # Enable foreign keys
                    cursor.execute("PRAGMA foreign_keys=ON")
                    # Set cache size (in pages, negative = KB)
                    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
                    # Set synchronous mode
                    cursor.execute("PRAGMA synchronous=NORMAL")
        
        @event.listens_for(self.engine, "close")
        def receive_close(dbapi_connection, connection_record):
            """Track closed connections"""
            self.stats['connections_closed'] += 1
            logger.debug("Database connection closed")
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Track query execution"""
            self.stats['queries_executed'] += 1
            context._query_start_time = time.time()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Executing query: {statement[:100]}...")
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries"""
            if hasattr(context, '_query_start_time'):
                execution_time = time.time() - context._query_start_time
                if execution_time > 1.0:  # Log slow queries (>1 second)
                    logger.warning(f"Slow query detected ({execution_time:.2f}s): {statement[:100]}...")
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with proper error handling and cleanup"""
        session = None
        try:
            session = self.SessionLocal()
            yield session
            session.commit()
            
        except SQLAlchemyError as e:
            if session:
                session.rollback()
            self.stats['errors_count'] += 1
            self.stats['last_error'] = str(e)
            logger.error(f"Database error: {e}")
            raise
            
        except Exception as e:
            if session:
                session.rollback()
            self.stats['errors_count'] += 1
            self.stats['last_error'] = str(e)
            logger.error(f"Unexpected database error: {e}")
            raise
            
        finally:
            if session:
                session.close()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query and return results"""
        with self.get_session() as session:
            try:
                result = session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result.fetchall()]
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
    
    def execute_query_with_retry(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute query with retry logic for connection issues"""
        last_exception = None
        
        for attempt in range(self._retry_attempts):
            try:
                return self.execute_query(query, params)
                
            except (DisconnectionError, OSError) as e:
                last_exception = e
                if attempt < self._retry_attempts - 1:
                    logger.warning(f"Database connection lost, retrying in {self._retry_delay}s... (attempt {attempt + 1}/{self._retry_attempts})")
                    time.sleep(self._retry_delay)
                    self._retry_delay *= 2  # Exponential backoff
                    continue
                break
                
            except Exception as e:
                # Don't retry for non-connection errors
                raise e
        
        # All retries failed
        logger.error(f"Query failed after {self._retry_attempts} attempts: {last_exception}")
        raise last_exception
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = time.time()
            
            # Test basic connectivity
            with self.get_session() as session:
                if self.database_url.startswith('sqlite'):
                    result = session.execute(text("SELECT 1"))
                else:  # PostgreSQL
                    result = session.execute(text("SELECT version()"))
                result.fetchone()
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time, 2),
                'database_type': self._get_database_type(),
                'uptime_seconds': (datetime.now() - self.stats['uptime_start']).total_seconds(),
                'statistics': self.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_type': self._get_database_type(),
                'statistics': self.get_statistics()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = datetime.now() - self.stats['uptime_start']
        
        return {
            'connections_created': self.stats['connections_created'],
            'connections_closed': self.stats['connections_closed'],
            'active_connections': self.stats['connections_created'] - self.stats['connections_closed'],
            'queries_executed': self.stats['queries_executed'],
            'errors_count': self.stats['errors_count'],
            'last_error': self.stats['last_error'],
            'uptime_seconds': uptime.total_seconds(),
            'queries_per_second': round(self.stats['queries_executed'] / uptime.total_seconds(), 2) if uptime.total_seconds() > 0 else 0
        }
    
    def optimize_database(self):
        """Perform database optimization tasks"""
        try:
            if self.database_url.startswith('sqlite'):
                self._optimize_sqlite()
            elif self.database_url.startswith('postgresql'):
                self._optimize_postgresql()
                
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    def _optimize_sqlite(self):
        """SQLite specific optimization"""
        with self.get_session() as session:
            # Analyze tables for better query planning
            session.execute(text("ANALYZE"))
            
            # Vacuum to reclaim space
            session.execute(text("VACUUM"))
            
            # Update table statistics
            session.execute(text("PRAGMA optimize"))
    
    def _optimize_postgresql(self):
        """PostgreSQL specific optimization"""
        with self.get_session() as session:
            # Update table statistics
            session.execute(text("ANALYZE"))
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            if self.database_url.startswith('sqlite'):
                return self._backup_sqlite(backup_path)
            elif self.database_url.startswith('postgresql'):
                return self._backup_postgresql(backup_path)
            else:
                logger.error("Backup not supported for this database type")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def _backup_sqlite(self, backup_path: str) -> bool:
        """Create SQLite backup"""
        import shutil
        
        try:
            # Extract database file path
            db_file = self.database_url.replace('sqlite:///', '')
            
            if os.path.exists(db_file):
                # Ensure backup directory exists
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Create backup with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = f"{backup_path}_{timestamp}.db"
                
                # Copy database file
                shutil.copy2(db_file, backup_file)
                
                logger.info(f"SQLite backup created: {backup_file}")
                return True
            else:
                logger.warning(f"Database file not found: {db_file}")
                return False
                
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            return False
    
    def _backup_postgresql(self, backup_path: str) -> bool:
        """Create PostgreSQL backup using pg_dump"""
        import subprocess
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{backup_path}_{timestamp}.sql"
            
            # Extract connection parameters from URL
            # postgresql://user:password@host:port/database
            url_parts = self.database_url.replace('postgresql://', '').split('/')
            connection_part = url_parts[0]
            database = url_parts[1] if len(url_parts) > 1 else 'postgres'
            
            if '@' in connection_part:
                auth_part, host_part = connection_part.split('@')
                username, password = auth_part.split(':') if ':' in auth_part else (auth_part, '')
            else:
                host_part = connection_part
                username, password = '', ''
            
            host, port = host_part.split(':') if ':' in host_part else (host_part, '5432')
            
            # Build pg_dump command
            cmd = [
                'pg_dump',
                '-h', host,
                '-p', port,
                '-U', username,
                '-d', database,
                '-f', backup_file,
                '--verbose'
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            if password:
                env['PGPASSWORD'] = password
            
            # Execute pg_dump
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"PostgreSQL backup created: {backup_file}")
                return True
            else:
                logger.error(f"pg_dump failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from database"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                # Clean old system logs
                deleted_logs = session.execute(
                    text("DELETE FROM system_logs WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                ).rowcount
                
                # Clean old AI analysis logs
                deleted_ai_logs = session.execute(
                    text("DELETE FROM ai_analysis_log WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                ).rowcount
                
                # Clean old market data (keep only last 90 days)
                market_cutoff = datetime.now() - timedelta(days=90)
                deleted_market_data = session.execute(
                    text("DELETE FROM market_data WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': market_cutoff}
                ).rowcount
                
                logger.info(f"Cleanup completed - Logs: {deleted_logs}, AI logs: {deleted_ai_logs}, Market data: {deleted_market_data}")
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def _get_database_type(self) -> str:
        """Get database type from URL"""
        if self.database_url.startswith('sqlite'):
            return 'SQLite'
        elif self.database_url.startswith('postgresql'):
            return 'PostgreSQL'
        elif self.database_url.startswith('mysql'):
            return 'MySQL'
        else:
            return 'Unknown'
    
    def _mask_credentials(self, url: str) -> str:
        """Mask credentials in database URL for logging"""
        if '://' in url and '@' in url:
            protocol, rest = url.split('://', 1)
            if '@' in rest:
                creds, host_db = rest.split('@', 1)
                return f"{protocol}://***:***@{host_db}"
        return url
    
    def close(self):
        """Close database connections and cleanup"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Global database manager instance
db_manager = None

def initialize_database_manager(database_url: str = None, echo: bool = False) -> DatabaseManager:
    """Initialize global database manager"""
    global db_manager
    
    if db_manager is None:
        db_manager = DatabaseManager(database_url, echo)
    
    return db_manager

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    
    if db_manager is None:
        db_manager = DatabaseManager()
    
    return db_manager

# Convenience functions
def get_db_session():
    """Get database session (convenience function)"""
    return get_database_manager().get_session()

def execute_db_query(query: str, params: Dict[str, Any] = None):
    """Execute database query (convenience function)"""
    return get_database_manager().execute_query(query, params)

def health_check_db():
    """Database health check (convenience function)"""
    return get_database_manager().health_check()

if __name__ == "__main__":
    # Test database connection
    print("🗄️  Database Connection Test")
    print("=" * 50)
    
    # Initialize database manager
    db_mgr = DatabaseManager()
    
    # Perform health check
    health = db_mgr.health_check()
    print(f"Database Status: {health['status']}")
    print(f"Database Type: {health['database_type']}")
    print(f"Response Time: {health.get('response_time_ms', 'N/A')} ms")
    
    if health['status'] == 'healthy':
        # Test basic operations
        try:
            # Test session context manager
            with db_mgr.get_session() as session:
                if health['database_type'] == 'SQLite':
                    result = session.execute(text("SELECT sqlite_version() as version"))
                else:
                    result = session.execute(text("SELECT version() as version"))
                
                version = result.fetchone()
                print(f"Database Version: {version[0] if version else 'Unknown'}")
            
            # Show statistics
            stats = db_mgr.get_statistics()
            print(f"Connections Created: {stats['connections_created']}")
            print(f"Queries Executed: {stats['queries_executed']}")
            
            print("\n✅ Database connection test successful!")
            
        except Exception as e:
            print(f"\n❌ Database test failed: {e}")
    else:
        print(f"\n❌ Database health check failed: {health.get('error', 'Unknown error')}")
    
    # Cleanup
    db_mgr.close()