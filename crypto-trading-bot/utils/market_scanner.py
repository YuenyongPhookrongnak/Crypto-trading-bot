import ccxt
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MarketScanResult:
    """Market scan result for a single symbol"""
    symbol: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    volume_ratio: float
    market_cap: float
    
    # Technical scores
    momentum_score: float
    volume_score: float
    volatility_score: float
    technical_score: float
    total_score: float
    
    # Market data
    high_24h: float
    low_24h: float
    support_level: float
    resistance_level: float
    
    # Indicators
    rsi: float
    macd: float
    sma_20: float
    sma_50: float
    
    # Metadata
    scan_timestamp: datetime = field(default_factory=datetime.utcnow)
    exchange: str = "binance"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'price_change_24h': self.price_change_24h,
            'volume_24h': self.volume_24h,
            'volume_ratio': self.volume_ratio,
            'market_cap': self.market_cap,
            'momentum_score': self.momentum_score,
            'volume_score': self.volume_score,
            'volatility_score': self.volatility_score,
            'technical_score': self.technical_score,
            'total_score': self.total_score,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'support_level': self.support_level,
            'resistance_level': self.resistance_level,
            'rsi': self.rsi,
            'macd': self.macd,
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'scan_timestamp': self.scan_timestamp.isoformat(),
            'exchange': self.exchange
        }

class MarketScanner:
    """Advanced Market Scanner for finding trading opportunities"""
    
    def __init__(self, api_config):
        self.api_config = api_config
        self.exchange = None
        
        # Scanning criteria
        self.scanning_criteria = {
            'min_volume_24h': 50_000_000,      # $50M minimum volume
            'min_price_change': 3.0,            # 3% minimum price change
            'volume_spike_threshold': 1.5,      # 1.5x volume spike
            'max_symbols_to_scan': 100,         # Maximum symbols to analyze
            'min_market_cap': 100_000_000,      # $100M minimum market cap
            'max_spread_percentage': 0.5,       # 0.5% maximum spread
            'min_liquidity_score': 60           # Minimum liquidity score
        }
        
        # State tracking
        self.last_scan_results = {}
        self.scan_history = []
        self.performance_metrics = {
            'total_scans': 0,
            'symbols_found': 0,
            'avg_scan_time': 0,
            'last_scan_time': None
        }
        
        # Cache for market data
        self.market_data_cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    async def initialize(self):
        """Initialize market scanner"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_config.binance_api_key,
                'secret': self.api_config.binance_secret,
                'sandbox': self.api_config.binance_testnet,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}  # Use spot market for scanning
            })
            
            await self.exchange.load_markets()
            logger.info("Market scanner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize market scanner: {e}")
            raise
    
    async def scan_for_opportunities(self, 
                                   custom_criteria: Dict[str, Any] = None,
                                   symbol_filter: List[str] = None) -> List[MarketScanResult]:
        """Scan market for trading opportunities"""
        
        start_time = time.time()
        
        try:
            # Update criteria if provided
            criteria = self.scanning_criteria.copy()
            if custom_criteria:
                criteria.update(custom_criteria)
            
            logger.info(f"Starting market scan with criteria: {criteria}")
            
            # Get all tickers
            tickers = await self.exchange.fetch_tickers()
            
            # Filter for USDT pairs
            usdt_pairs = self._filter_usdt_pairs(tickers, symbol_filter)
            logger.info(f"Found {len(usdt_pairs)} USDT pairs to analyze")
            
            # Analyze symbols concurrently
            scan_tasks = []
            for symbol, ticker in usdt_pairs.items():
                task = self._analyze_symbol(symbol, ticker, criteria)
                scan_tasks.append(task)
            
            # Execute analysis
            if scan_tasks:
                results = await asyncio.gather(*scan_tasks, return_exceptions=True)
                
                # Process results
                valid_results = []
                for result in results:
                    if isinstance(result, MarketScanResult):
                        valid_results.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Analysis error: {result}")
                
                # Sort by total score
                valid_results.sort(key=lambda x: x.total_score, reverse=True)
                
                # Limit results
                max_results = criteria.get('max_symbols_to_scan', 50)
                top_results = valid_results[:max_results]
                
                # Update performance metrics
                scan_time = time.time() - start_time
                self._update_performance_metrics(len(top_results), scan_time)
                
                # Store results
                self.last_scan_results = {
                    'timestamp': datetime.utcnow(),
                    'total_symbols_scanned': len(usdt_pairs),
                    'opportunities_found': len(top_results),
                    'top_opportunities': top_results,
                    'scan_duration': scan_time
                }
                
                # Add to history
                self.scan_history.append(self.last_scan_results)
                if len(self.scan_history) > 100:  # Keep last 100 scans
                    self.scan_history = self.scan_history[-100:]
                
                logger.info(f"Market scan completed: {len(top_results)} opportunities found in {scan_time:.2f}s")
                return top_results
            
            return []
            
        except Exception as e:
            logger.error(f"Error scanning market: {e}")
            return []
    
    def _filter_usdt_pairs(self, 
                          tickers: Dict[str, Any], 
                          symbol_filter: List[str] = None) -> Dict[str, Any]:
        """Filter for valid USDT trading pairs"""
        
        usdt_pairs = {}
        
        for symbol, ticker in tickers.items():
            # Must be USDT pair
            if not symbol.endswith('/USDT'):
                continue
            
            # Apply symbol filter if provided
            if symbol_filter and symbol not in symbol_filter:
                continue
            
            # Skip if invalid symbol
            if not self._is_valid_symbol(symbol):
                continue
            
            # Basic validation
            if not ticker or not ticker.get('last') or not ticker.get('quoteVolume'):
                continue
            
            # Volume check
            volume_24h = ticker.get('quoteVolume', 0)
            if volume_24h < self.scanning_criteria['min_volume_24h']:
                continue
            
            usdt_pairs[symbol] = ticker
        
        return usdt_pairs
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for scanning"""
        
        # Skip stablecoins
        excluded_bases = ['USDC/USDT', 'BUSD/USDT', 'TUSD/USDT', 'USTC/USDT', 'DAI/USDT']
        if symbol in excluded_bases:
            return False
        
        # Skip leverage tokens
        leverage_patterns = ['UP/', 'DOWN/', 'BULL/', 'BEAR/', '3L/', '3S/']
        if any(pattern in symbol for pattern in leverage_patterns):
            return False
        
        # Skip very new or delisted coins (basic check)
        base_currency = symbol.replace('/USDT', '')
        if len(base_currency) > 10:  # Likely a complex token name
            return False
        
        return True
    
    async def _analyze_symbol(self, 
                             symbol: str, 
                             ticker: Dict[str, Any], 
                             criteria: Dict[str, Any]) -> Optional[MarketScanResult]:
        """Analyze individual symbol for opportunities"""
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Basic ticker analysis
            current_price = float(ticker.get('last', 0))
            price_change_24h = float(ticker.get('percentage', 0))
            volume_24h = float(ticker.get('quoteVolume', 0))
            high_24h = float(ticker.get('high', current_price))
            low_24h = float(ticker.get('low', current_price))
            
            if current_price <= 0 or volume_24h <= 0:
                return None
            
            # Get OHLCV data for technical analysis
            ohlcv_data = await self._get_ohlcv_data(symbol)
            if not ohlcv_data or len(ohlcv_data) < 20:
                logger.debug(f"Insufficient OHLCV data for {symbol}")
                return None
            
            # Calculate scores
            momentum_score = self._calculate_momentum_score(ticker, ohlcv_data)
            volume_score = self._calculate_volume_score(ticker, ohlcv_data)
            volatility_score = self._calculate_volatility_score(ticker, ohlcv_data)
            technical_score = self._calculate_technical_score(ohlcv_data)
            
            # Calculate total score
            total_score = (momentum_score * 0.3 + 
                          volume_score * 0.25 + 
                          volatility_score * 0.2 + 
                          technical_score * 0.25)
            
            # Filter by minimum score
            min_score = criteria.get('min_total_score', 50)
            if total_score < min_score:
                return None
            
            # Calculate additional metrics
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Technical indicators
            rsi = self._calculate_rsi(df['close'])
            macd = self._calculate_macd(df['close'])
            sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else current_price
            
            # Support/Resistance levels
            support_level = df['low'].rolling(20).min().iloc[-1] if len(df) >= 20 else low_24h
            resistance_level = df['high'].rolling(20).max().iloc[-1] if len(df) >= 20 else high_24h
            
            # Volume ratio
            avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else volume_24h
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
            
            # Estimate market cap (rough calculation)
            base_currency = symbol.replace('/USDT', '')
            market_cap = await self._estimate_market_cap(base_currency, current_price, volume_24h)
            
            # Create result
            result = MarketScanResult(
                symbol=symbol,
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                volume_ratio=volume_ratio,
                market_cap=market_cap,
                momentum_score=momentum_score,
                volume_score=volume_score,
                volatility_score=volatility_score,
                technical_score=technical_score,
                total_score=total_score,
                high_24h=high_24h,
                low_24h=low_24h,
                support_level=support_level,
                resistance_level=resistance_level,
                rsi=rsi,
                macd=macd,
                sma_20=sma_20,
                sma_50=sma_50
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
            return None
    
    async def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _get_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List]:
        """Get OHLCV data with caching"""
        
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.market_data_cache:
            cached_data, cache_time = self.market_data_cache[cache_key]
            if current_time - cache_time < self.cache_duration:
                return cached_data
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Cache the data
            self.market_data_cache[cache_key] = (ohlcv, current_time)
            
            return ohlcv
            
        except Exception as e:
            logger.warning(f"Error fetching OHLCV for {symbol}: {e}")
            return None
    
    def _calculate_momentum_score(self, ticker: Dict[str, Any], ohlcv_data: List) -> float:
        """Calculate momentum score (0-100)"""
        
        try:
            price_change = abs(float(ticker.get('percentage', 0)))
            
            # Price change component
            if price_change >= 10:
                momentum_score = 100
            elif price_change >= 5:
                momentum_score = 80
            elif price_change >= 3:
                momentum_score = 60
            elif price_change >= 1:
                momentum_score = 40
            else:
                momentum_score = 20
            
            # Additional momentum from OHLCV data
            if len(ohlcv_data) >= 10:
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Price momentum over different periods
                current_price = df['close'].iloc[-1]
                price_6h_ago = df['close'].iloc[-6] if len(df) >= 6 else current_price
                price_24h_ago = df['close'].iloc[-24] if len(df) >= 24 else current_price
                
                momentum_6h = (current_price - price_6h_ago) / price_6h_ago if price_6h_ago > 0 else 0
                momentum_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0
                
                # Adjust score based on sustained momentum
                if momentum_6h > 0 and momentum_24h > 0:
                    momentum_score *= 1.2  # Boost for sustained upward momentum
                elif momentum_6h < 0 and momentum_24h < 0:
                    momentum_score *= 1.2  # Boost for sustained downward momentum
            
            return min(100, momentum_score)
            
        except Exception as e:
            logger.warning(f"Error calculating momentum score: {e}")
            return 0
    
    def _calculate_volume_score(self, ticker: Dict[str, Any], ohlcv_data: List) -> float:
        """Calculate volume score (0-100)"""
        
        try:
            current_volume = float(ticker.get('quoteVolume', 0))
            
            if len(ohlcv_data) >= 20:
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    
                    if volume_ratio >= 3.0:
                        return 100
                    elif volume_ratio >= 2.0:
                        return 80
                    elif volume_ratio >= 1.5:
                        return 60
                    elif volume_ratio >= 1.0:
                        return 40
                    else:
                        return 20
            
            # Fallback: score based on absolute volume
            if current_volume >= 500_000_000:  # $500M+
                return 100
            elif current_volume >= 200_000_000:  # $200M+
                return 80
            elif current_volume >= 100_000_000:  # $100M+
                return 60
            elif current_volume >= 50_000_000:   # $50M+
                return 40
            else:
                return 20
                
        except Exception as e:
            logger.warning(f"Error calculating volume score: {e}")
            return 0
    
    def _calculate_volatility_score(self, ticker: Dict[str, Any], ohlcv_data: List) -> float:
        """Calculate volatility score (0-100)"""
        
        try:
            # Price range volatility
            high_24h = float(ticker.get('high', 0))
            low_24h = float(ticker.get('low', 0))
            current_price = float(ticker.get('last', 0))
            
            if current_price > 0:
                price_range = (high_24h - low_24h) / current_price
                
                if price_range >= 0.15:  # 15%+
                    volatility_score = 100
                elif price_range >= 0.10:  # 10%+
                    volatility_score = 80
                elif price_range >= 0.05:  # 5%+
                    volatility_score = 60
                elif price_range >= 0.02:  # 2%+
                    volatility_score = 40
                else:
                    volatility_score = 20
                
                # Adjust based on historical volatility
                if len(ohlcv_data) >= 20:
                    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    returns = df['close'].pct_change().dropna()
                    historical_volatility = returns.std()
                    
                    if historical_volatility > 0.05:  # High historical volatility
                        volatility_score *= 1.1
                
                return min(100, volatility_score)
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error calculating volatility score: {e}")
            return 0
    
    def _calculate_technical_score(self, ohlcv_data: List) -> float:
        """Calculate technical analysis score (0-100)"""
        
        try:
            if len(ohlcv_data) < 20:
                return 0
            
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            technical_score = 0
            factors = 0
            
            # RSI score
            rsi = self._calculate_rsi(df['close'])
            if 20 <= rsi <= 30 or 70 <= rsi <= 80:  # Extreme levels
                technical_score += 25
                factors += 1
            elif 30 <= rsi <= 40 or 60 <= rsi <= 70:  # Moderate levels
                technical_score += 15
                factors += 1
            elif 40 <= rsi <= 60:  # Neutral
                technical_score += 5
                factors += 1
            
            # Moving average alignment
            if len(df) >= 50:
                sma_20 = df['close'].rolling(20).mean().iloc[-1]
                sma_50 = df['close'].rolling(50).mean().iloc[-1]
                current_price = df['close'].iloc[-1]
                
                # Bullish alignment
                if current_price > sma_20 > sma_50:
                    technical_score += 20
                # Bearish alignment
                elif current_price < sma_20 < sma_50:
                    technical_score += 20
                # Mixed signals
                else:
                    technical_score += 5
                factors += 1
            
            # MACD signal
            macd = self._calculate_macd(df['close'])
            if abs(macd) > 0:  # Strong MACD signal
                technical_score += 15
            else:
                technical_score += 5
            factors += 1
            
            # Volume-price relationship
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-5]) / df['volume'].iloc[-5]
            
            if (price_change > 0 and volume_change > 0) or (price_change < 0 and volume_change > 0):
                technical_score += 15  # Volume confirms price movement
            else:
                technical_score += 5
            factors += 1
            
            # Average the scores
            return technical_score / factors if factors > 0 else 0
            
        except Exception as e:
            logger.warning(f"Error calculating technical score: {e}")
            return 0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        except:
            return 50
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """Calculate MACD indicator"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            return macd.iloc[-1] if not macd.empty else 0
        except:
            return 0
    
    async def _estimate_market_cap(self, base_currency: str, price: float, volume_24h: float) -> float:
        """Estimate market cap (simplified calculation)"""
        try:
            # This is a simplified estimation
            # In a real implementation, you would fetch actual supply data
            
            # Rough supply estimates for common cryptocurrencies
            supply_estimates = {
                'BTC': 19_000_000,
                'ETH': 120_000_000,
                'BNB': 150_000_000,
                'ADA': 35_000_000_000,
                'DOT': 1_000_000_000,
                'SOL': 400_000_000,
                'MATIC': 9_000_000_000,
                'LINK': 500_000_000,
                'UNI': 750_000_000,
                'AAVE': 16_000_000
            }
            
            # Estimate supply based on volume (very rough)
            estimated_supply = supply_estimates.get(base_currency, volume_24h / price * 1000)
            
            return estimated_supply * price
            
        except:
            return 0
    
    def _update_performance_metrics(self, symbols_found: int, scan_time: float):
        """Update scanner performance metrics"""
        
        self.performance_metrics['total_scans'] += 1
        self.performance_metrics['symbols_found'] += symbols_found
        
        # Update average scan time
        total_scans = self.performance_metrics['total_scans']
        current_avg = self.performance_metrics['avg_scan_time']
        self.performance_metrics['avg_scan_time'] = (current_avg * (total_scans - 1) + scan_time) / total_scans
        
        self.performance_metrics['last_scan_time'] = datetime.utcnow()
    
    async def get_top_gainers(self, limit: int = 20) -> List[MarketScanResult]:
        """Get top gaining cryptocurrencies"""
        
        criteria = {
            'min_price_change': 0,  # No minimum change
            'max_symbols_to_scan': limit * 2  # Scan more to get top gainers
        }
        
        results = await self.scan_for_opportunities(criteria)
        
        # Sort by price change
        gainers = sorted(results, key=lambda x: x.price_change_24h, reverse=True)
        
        return gainers[:limit]
    
    async def get_top_losers(self, limit: int = 20) -> List[MarketScanResult]:
        """Get top losing cryptocurrencies"""
        
        criteria = {
            'min_price_change': 0,  # No minimum change
            'max_symbols_to_scan': limit * 2
        }
        
        results = await self.scan_for_opportunities(criteria)
        
        # Sort by price change (ascending for losers)
        losers = sorted(results, key=lambda x: x.price_change_24h)
        
        return losers[:limit]
    
    async def get_high_volume_opportunities(self, limit: int = 20) -> List[MarketScanResult]:
        """Get opportunities with highest volume"""
        
        criteria = {
            'min_volume_24h': 100_000_000,  # Higher volume threshold
            'max_symbols_to_scan': limit * 2
        }
        
        results = await self.scan_for_opportunities(criteria)
        
        # Sort by volume
        high_volume = sorted(results, key=lambda x: x.volume_24h, reverse=True)
        
        return high_volume[:limit]
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scanner performance statistics"""
        
        stats = self.performance_metrics.copy()
        
        if self.last_scan_results:
            stats.update({
                'last_scan_symbols': self.last_scan_results['total_symbols_scanned'],
                'last_scan_opportunities': self.last_scan_results['opportunities_found'],
                'last_scan_duration': self.last_scan_results['scan_duration'],
                'last_scan_timestamp': self.last_scan_results['timestamp'].isoformat()
            })
        
        # Calculate average opportunities per scan
        if stats['total_scans'] > 0:
            stats['avg_opportunities_per_scan'] = stats['symbols_found'] / stats['total_scans']
        else:
            stats['avg_opportunities_per_scan'] = 0
        
        return stats
    
    def get_scan_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scan history"""
        
        history = []
        for scan in self.scan_history[-limit:]:
            history.append({
                'timestamp': scan['timestamp'].isoformat(),
                'symbols_scanned': scan['total_symbols_scanned'],
                'opportunities_found': scan['opportunities_found'],
                'scan_duration': scan['scan_duration'],
                'top_symbol': scan['top_opportunities'][0].symbol if scan['top_opportunities'] else None,
                'top_score': scan['top_opportunities'][0].total_score if scan['top_opportunities'] else 0
            })
        
        return history
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        
        try:
            # Test exchange connectivity
            await self.exchange.fetch_status()
            
            # Test API limits
            tickers = await self.exchange.fetch_tickers(['BTC/USDT'])
            
            return {
                'status': 'healthy',
                'exchange_connected': True,
                'api_responsive': True,
                'cache_size': len(self.market_data_cache),
                'last_scan': self.performance_metrics['last_scan_time'].isoformat() if self.performance_metrics['last_scan_time'] else None
            }
            
        except Exception as e:
            logger.error(f"Market scanner health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'exchange_connected': False,
                'api_responsive': False
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        
        try:
            if self.exchange:
                await self.exchange.close()
            
            # Clear cache
            self.market_data_cache.clear()
            
            logger.info("Market scanner cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during market scanner cleanup: {e}")

# Factory function
def create_market_scanner(api_config) -> MarketScanner:
    """Factory function to create Market Scanner instance"""
    return MarketScanner(api_config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Mock API config for testing
    class MockApiConfig:
        def __init__(self):
            self.binance_api_key = "test_api_key"
            self.binance_secret = "test_secret"
            self.binance_testnet = True
    
    async def test_market_scanner():
        print("🔍 Market Scanner Test")
        print("=" * 50)
        
        try:
            # Create mock scanner (would normally use real API config)
            api_config = MockApiConfig()
            scanner = create_market_scanner(api_config)
            
            print(f"✅ Scanner created: {scanner}")
            print(f"📊 Scanning criteria: {scanner.scanning_criteria}")
            
            # Test scanning criteria validation
            print("\n🎯 Testing scanning criteria...")
            test_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT']
            
            for symbol in test_symbols:
                is_valid = scanner._is_valid_symbol(symbol)
                print(f"  {symbol}: {'✅ Valid' if is_valid else '❌ Invalid'}")
            
            # Test invalid symbols
            invalid_symbols = ['USDC/USDT', 'ETHUP/USDT', 'BTCDOWN/USDT']
            for symbol in invalid_symbols:
                is_valid = scanner._is_valid_symbol(symbol)
                print(f"  {symbol}: {'✅ Valid' if is_valid else '❌ Invalid'}")
            
            # Test technical calculations with sample data
            print("\n📈 Testing technical calculations...")
            
            # Create sample OHLCV data
            dates = pd.date_range('2024-01-01', periods=100, freq='1H')
            np.random.seed(42)
            
            base_price = 50000
            prices = [base_price]
            volumes = []
            
            for i in range(99):
                # Random walk with slight upward bias
                change = np.random.normal(100, 500)  # Mean: +$100, Std: $500
                new_price = max(prices[-1] + change, base_price * 0.8)
                prices.append(new_price)
                
                # Volume with occasional spikes
                base_vol = np.random.lognormal(15, 0.3)
                if np.random.random() < 0.1:  # 10% chance of volume spike
                    base_vol *= 3
                volumes.append(base_vol)
            
            # Create sample ticker data
            current_price = prices[-1]
            price_change_24h = ((current_price - prices[-24]) / prices[-24]) * 100
            
            sample_ticker = {
                'last': current_price,
                'percentage': price_change_24h,
                'quoteVolume': volumes[-1],
                'high': max(prices[-24:]),
                'low': min(prices[-24:])
            }
            
            # Create OHLCV data
            ohlcv_data = []
            for i in range(len(prices)):
                high = prices[i] + abs(np.random.normal(0, 200))
                low = prices[i] - abs(np.random.normal(0, 200))
                ohlcv_data.append([
                    dates[i].timestamp() * 1000,  # timestamp
                    prices[i] + np.random.normal(0, 50),  # open
                    high,  # high
                    low,   # low
                    prices[i],  # close
                    volumes[i] if i < len(volumes) else volumes[-1]  # volume
                ])
            
            print(f"  Sample data: {len(ohlcv_data)} periods")
            print(f"  Current price: ${current_price:,.2f}")
            print(f"  24h change: {price_change_24h:+.2f}%")
            print(f"  Volume: ${volumes[-1]:,.0f}")
            
            # Test scoring functions
            momentum_score = scanner._calculate_momentum_score(sample_ticker, ohlcv_data)
            volume_score = scanner._calculate_volume_score(sample_ticker, ohlcv_data)
            volatility_score = scanner._calculate_volatility_score(sample_ticker, ohlcv_data)
            technical_score = scanner._calculate_technical_score(ohlcv_data)
            
            print(f"\n📊 Scoring results:")
            print(f"  Momentum Score: {momentum_score:.1f}/100")
            print(f"  Volume Score: {volume_score:.1f}/100")
            print(f"  Volatility Score: {volatility_score:.1f}/100")
            print(f"  Technical Score: {technical_score:.1f}/100")
            
            total_score = (momentum_score * 0.3 + 
                          volume_score * 0.25 + 
                          volatility_score * 0.2 + 
                          technical_score * 0.25)
            print(f"  Total Score: {total_score:.1f}/100")
            
            # Test technical indicators
            print(f"\n🔍 Technical indicators:")
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            rsi = scanner._calculate_rsi(df['close'])
            macd = scanner._calculate_macd(df['close'])
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            print(f"  RSI: {rsi:.1f}")
            print(f"  MACD: {macd:.2f}")
            print(f"  SMA 20: ${sma_20:,.2f}")
            print(f"  SMA 50: ${sma_50:,.2f}")
            
            # Test market cap estimation
            market_cap = await scanner._estimate_market_cap('BTC', current_price, volumes[-1])
            print(f"  Estimated Market Cap: ${market_cap:,.0f}")
            
            # Create a complete scan result for demonstration
            scan_result = MarketScanResult(
                symbol='BTC/USDT',
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volumes[-1],
                volume_ratio=volumes[-1] / np.mean(volumes[-20:]),
                market_cap=market_cap,
                momentum_score=momentum_score,
                volume_score=volume_score,
                volatility_score=volatility_score,
                technical_score=technical_score,
                total_score=total_score,
                high_24h=sample_ticker['high'],
                low_24h=sample_ticker['low'],
                support_level=df['low'].rolling(20).min().iloc[-1],
                resistance_level=df['high'].rolling(20).max().iloc[-1],
                rsi=rsi,
                macd=macd,
                sma_20=sma_20,
                sma_50=sma_50
            )
            
            print(f"\n📋 Complete scan result:")
            result_dict = scan_result.to_dict()
            for key, value in result_dict.items():
                if isinstance(value, float):
                    if key.endswith('_score') or key == 'rsi':
                        print(f"  {key}: {value:.1f}")
                    elif 'price' in key or 'level' in key or key.startswith('sma'):
                        print(f"  {key}: ${value:,.2f}")
                    elif 'volume' in key or 'market_cap' in key:
                        print(f"  {key}: {value:,.0f}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Test performance metrics
            print(f"\n📈 Performance metrics:")
            scanner._update_performance_metrics(1, 2.5)  # 1 symbol found, 2.5 seconds
            scanner._update_performance_metrics(3, 1.8)  # 3 symbols found, 1.8 seconds
            
            stats = scanner.get_scan_statistics()
            print(f"  Total scans: {stats['total_scans']}")
            print(f"  Symbols found: {stats['symbols_found']}")
            print(f"  Average scan time: {stats['avg_scan_time']:.2f}s")
            print(f"  Average opportunities per scan: {stats['avg_opportunities_per_scan']:.1f}")
            
            # Test criteria validation
            print(f"\n⚙️ Testing custom criteria:")
            custom_criteria = {
                'min_volume_24h': 100_000_000,  # $100M
                'min_price_change': 5.0,          # 5%
                'max_symbols_to_scan': 20
            }
            
            print(f"  Custom criteria: {custom_criteria}")
            
            # Simulate filtering with criteria
            meets_volume = volumes[-1] >= custom_criteria['min_volume_24h']
            meets_change = abs(price_change_24h) >= custom_criteria['min_price_change']
            meets_score = total_score >= 50  # Default minimum score
            
            print(f"  Meets volume criteria: {'✅' if meets_volume else '❌'} (${volumes[-1]:,.0f})")
            print(f"  Meets change criteria: {'✅' if meets_change else '❌'} ({price_change_24h:+.2f}%)")
            print(f"  Meets score criteria: {'✅' if meets_score else '❌'} ({total_score:.1f})")
            
            overall_pass = meets_volume and meets_change and meets_score
            print(f"  Overall qualification: {'✅ PASS' if overall_pass else '❌ FAIL'}")
            
            print(f"\n🎉 Market Scanner test completed successfully!")
            
            # Test different scan types
            print(f"\n🔍 Testing scan type methods:")
            
            # Mock some results for testing
            mock_results = [
                MarketScanResult('BTC/USDT', 50000, 8.5, 200000000, 1.8, 950000000000, 85, 90, 75, 80, 82.5, 51000, 49000, 48500, 51500, 65, 150, 49800, 49200),
                MarketScanResult('ETH/USDT', 3000, -6.2, 150000000, 2.1, 360000000000, 70, 85, 80, 75, 77.5, 3100, 2900, 2850, 3150, 35, -80, 2980, 2950),
                MarketScanResult('ADA/USDT', 0.45, 12.3, 80000000, 3.2, 15000000000, 90, 95, 85, 70, 85.0, 0.48, 0.42, 0.41, 0.49, 75, 220, 0.44, 0.43)
            ]
            
            # Sort by different criteria
            top_gainers = sorted(mock_results, key=lambda x: x.price_change_24h, reverse=True)
            top_losers = sorted(mock_results, key=lambda x: x.price_change_24h)
            high_volume = sorted(mock_results, key=lambda x: x.volume_24h, reverse=True)
            
            print(f"  Top Gainer: {top_gainers[0].symbol} ({top_gainers[0].price_change_24h:+.1f}%)")
            print(f"  Top Loser: {top_losers[0].symbol} ({top_losers[0].price_change_24h:+.1f}%)")
            print(f"  Highest Volume: {high_volume[0].symbol} (${high_volume[0].volume_24h:,.0f})")
            
        except Exception as e:
            print(f"❌ Error in Market Scanner test: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Market Scanner test completed!")
    
    # Run the test
    asyncio.run(test_market_scanner())