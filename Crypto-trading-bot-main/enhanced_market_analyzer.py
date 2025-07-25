#!/usr/bin/env python3
"""
Enhanced Market Analysis with Additional Technical Tools

เพิ่ม Technical Indicators เพิ่มเติม:
- OBV (On-Balance Volume)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- Bollinger Bands Width
- Williams %R
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import asyncio

class EnhancedTechnicalAnalysis:
    """Enhanced Technical Analysis with additional indicators"""
    
    def __init__(self):
        self.indicators = {
            'OBV': self.calculate_obv,
            'ATR': self.calculate_atr,
            'VWAP': self.calculate_vwap,
            'BB_WIDTH': self.calculate_bb_width,
            'WILLIAMS_R': self.calculate_williams_r,
            'CMF': self.calculate_cmf,  # Chaikin Money Flow
            'STOCH_RSI': self.calculate_stoch_rsi
        }
        
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV)"""
        try:
            obv = []
            obv_value = 0
            
            for i in range(len(df)):
                if i == 0:
                    obv.append(df['volume'].iloc[i])
                    obv_value = df['volume'].iloc[i]
                else:
                    if df['close'].iloc[i] > df['close'].iloc[i-1]:
                        obv_value += df['volume'].iloc[i]
                    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                        obv_value -= df['volume'].iloc[i]
                    obv.append(obv_value)
            
            return pd.Series(obv, index=df.index)
        except Exception as e:
            print(f"Error calculating OBV: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(0)
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return vwap.fillna(df['close'])
        except Exception as e:
            print(f"Error calculating VWAP: {e}")
            return df['close']
    
    def calculate_bb_width(self, df: pd.DataFrame, period: int = 20, std: float = 2) -> pd.Series:
        """Calculate Bollinger Bands Width"""
        try:
            sma = df['close'].rolling(window=period).mean()
            std_dev = df['close'].rolling(window=period).std()
            
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            
            bb_width = (upper_band - lower_band) / sma * 100
            return bb_width.fillna(0)
        except Exception as e:
            print(f"Error calculating BB Width: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            return williams_r.fillna(-50)
        except Exception as e:
            print(f"Error calculating Williams %R: {e}")
            return pd.Series([-50] * len(df), index=df.index)
    
    def calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow (CMF)"""
        try:
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mf_volume = mf_multiplier * df['volume']
            
            cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
            return cmf.fillna(0)
        except Exception as e:
            print(f"Error calculating CMF: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_stoch_rsi(self, df: pd.DataFrame, period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic RSI"""
        try:
            # Calculate RSI first
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate Stochastic of RSI
            lowest_rsi = rsi.rolling(window=period).min()
            highest_rsi = rsi.rolling(window=period).max()
            
            stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
            stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean()
            stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
            
            return stoch_rsi_k.fillna(50), stoch_rsi_d.fillna(50)
        except Exception as e:
            print(f"Error calculating Stochastic RSI: {e}")
            return pd.Series([50] * len(df), index=df.index), pd.Series([50] * len(df), index=df.index)

class EnhancedSignalStrengthCalculator:
    """คำนวณ Signal Strength จากหลาย indicators แบบถ่วงน้ำหนัก"""
    
    def __init__(self):
        self.weights = {
            'BOS_COC': 25,      # Break of Structure/Change of Character
            'FVG': 20,          # Fair Value Gap
            'EMA_ALIGNMENT': 15, # EMA Alignment
            'RSI': 10,          # RSI conditions
            'VOLUME': 10,       # Volume confirmation
            'ATR': 5,           # Volatility consideration
            'OBV': 5,           # On-Balance Volume
            'VWAP': 5,          # VWAP position
            'NEWS_SENTIMENT': 5  # News sentiment
        }
        
    def calculate_signal_strength(self, analysis_data: Dict[str, Any]) -> float:
        """คำนวณ Signal Strength แบบถ่วงน้ำหนัก"""
        total_score = 0
        max_possible_score = sum(self.weights.values())
        
        # 1. BOS/CoC Analysis (25 points)
        if analysis_data.get('bos_bullish') or analysis_data.get('bos_bearish'):
            total_score += self.weights['BOS_COC']
        elif analysis_data.get('momentum_change'):
            total_score += self.weights['BOS_COC'] * 0.6  # Partial score
        
        # 2. FVG Analysis (20 points)
        fvgs = analysis_data.get('fvgs', [])
        current_price = analysis_data.get('current_price', 0)
        
        fvg_score = 0
        for fvg in fvgs[-3:]:  # Check last 3 FVGs
            if self._price_in_fvg(current_price, fvg):
                fvg_score = self.weights['FVG']
                break
            elif self._price_near_fvg(current_price, fvg, 0.005):  # Within 0.5%
                fvg_score = self.weights['FVG'] * 0.7
                break
        
        total_score += fvg_score
        
        # 3. EMA Alignment (15 points)
        if analysis_data.get('bullish_alignment') or analysis_data.get('bearish_alignment'):
            total_score += self.weights['EMA_ALIGNMENT']
        elif analysis_data.get('price_above_ema20'):
            total_score += self.weights['EMA_ALIGNMENT'] * 0.5
        
        # 4. RSI Analysis (10 points)
        rsi = analysis_data.get('rsi', 50)
        if rsi > 70 or rsi < 30:  # Extreme levels
            total_score += self.weights['RSI']
        elif rsi > 60 or rsi < 40:  # Moderate levels
            total_score += self.weights['RSI'] * 0.6
        
        # 5. Volume Analysis (10 points)
        volume_ratio = analysis_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            total_score += self.weights['VOLUME']
        elif volume_ratio > 1.2:
            total_score += self.weights['VOLUME'] * 0.7
        
        # 6. ATR Analysis (5 points) - Volatility consideration
        atr_score = analysis_data.get('atr_score', 0)
        total_score += min(atr_score, self.weights['ATR'])
        
        # 7. OBV Analysis (5 points)
        obv_score = analysis_data.get('obv_score', 0)
        total_score += min(obv_score, self.weights['OBV'])
        
        # 8. VWAP Analysis (5 points)
        vwap_score = analysis_data.get('vwap_score', 0)
        total_score += min(vwap_score, self.weights['VWAP'])
        
        # 9. News Sentiment (5 points)
        sentiment = analysis_data.get('sentiment_score', 0)
        if abs(sentiment) >= 5:
            total_score += self.weights['NEWS_SENTIMENT']
        elif abs(sentiment) >= 3:
            total_score += self.weights['NEWS_SENTIMENT'] * 0.6
        
        # Calculate percentage
        signal_strength = (total_score / max_possible_score) * 100
        
        return min(signal_strength, 95)  # Cap at 95%
    
    def _price_in_fvg(self, price: float, fvg: Dict[str, Any]) -> bool:
        """Check if price is within FVG zone"""
        try:
            bottom = fvg.get('bottom', 0)
            top = fvg.get('top', 0)
            return bottom <= price <= top
        except:
            return False
    
    def _price_near_fvg(self, price: float, fvg: Dict[str, Any], threshold: float) -> bool:
        """Check if price is near FVG zone"""
        try:
            bottom = fvg.get('bottom', 0)
            top = fvg.get('top', 0)
            zone_center = (bottom + top) / 2
            distance = abs(price - zone_center) / price
            return distance <= threshold
        except:
            return False

class EnhancedMarketAnalyzer:
    """Enhanced Market Analyzer with additional technical tools"""
    
    def __init__(self):
        self.tech_analysis = EnhancedTechnicalAnalysis()
        self.signal_calculator = EnhancedSignalStrengthCalculator()
        
    async def comprehensive_enhanced_analysis(self, exchange: ccxt.Exchange, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Comprehensive analysis with enhanced technical indicators"""
        try:
            # Fetch market data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate all technical indicators
            analysis_data = await self._calculate_all_indicators(df, symbol)
            
            # Calculate enhanced signal strength
            signal_strength = self.signal_calculator.calculate_signal_strength(analysis_data)
            analysis_data['signal_strength'] = signal_strength
            
            # Generate detailed confluence analysis
            confluences = self._generate_enhanced_confluences(analysis_data)
            analysis_data['confluences'] = confluences
            analysis_data['confluence_count'] = len(confluences)
            
            return analysis_data
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis: {e}")
            return {}
    
    async def _calculate_all_indicators(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate all technical indicators including new ones"""
        current_price = df['close'].iloc[-1]
        
        # Basic indicators
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Enhanced technical indicators
        df['obv'] = self.tech_analysis.calculate_obv(df)
        df['atr'] = self.tech_analysis.calculate_atr(df)
        df['vwap'] = self.tech_analysis.calculate_vwap(df)
        df['bb_width'] = self.tech_analysis.calculate_bb_width(df)
        df['williams_r'] = self.tech_analysis.calculate_williams_r(df)
        df['cmf'] = self.tech_analysis.calculate_cmf(df)
        df['stoch_rsi_k'], df['stoch_rsi_d'] = self.tech_analysis.calculate_stoch_rsi(df)
        
        # Current values
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current['volume'] / avg_volume if avg_volume > 0 else 1.0
        
        # OBV trend analysis
        obv_trend = "BULLISH" if current['obv'] > previous['obv'] else "BEARISH"
        obv_score = 5 if (
            (obv_trend == "BULLISH" and current_price > previous['close']) or
            (obv_trend == "BEARISH" and current_price < previous['close'])
        ) else 2
        
        # ATR-based volatility analysis
        atr_pct = current['atr'] / current_price * 100
        atr_score = 5 if 1.0 <= atr_pct <= 3.0 else 2  # Optimal volatility range
        
        # VWAP analysis
        vwap_position = "ABOVE" if current_price > current['vwap'] else "BELOW"
        vwap_score = 5 if (
            (vwap_position == "ABOVE" and current['rsi'] < 70) or
            (vwap_position == "BELOW" and current['rsi'] > 30)
        ) else 2
        
        # Structure analysis
        bos_analysis = self._detect_enhanced_bos(df)
        fvgs = self._identify_enhanced_fvg(df)
        
        return {
            'symbol': symbol,
            'timeframe': '1h',
            'current_price': current_price,
            'price_change_24h': ((current_price - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100 if len(df) >= 24 else 0,
            
            # Basic indicators
            'rsi': current['rsi'],
            'ema20': current['ema20'],
            'ema50': current['ema50'],
            'ema200': current['ema200'],
            
            # Enhanced indicators
            'obv': current['obv'],
            'obv_trend': obv_trend,
            'obv_score': obv_score,
            'atr': current['atr'],
            'atr_pct': atr_pct,
            'atr_score': atr_score,
            'vwap': current['vwap'],
            'vwap_position': vwap_position,
            'vwap_score': vwap_score,
            'bb_width': current['bb_width'],
            'williams_r': current['williams_r'],
            'cmf': current['cmf'],
            'stoch_rsi_k': current['stoch_rsi_k'],
            'stoch_rsi_d': current['stoch_rsi_d'],
            
            # Volume analysis
            'volume': current['volume'],
            'volume_ratio': volume_ratio,
            
            # Structure analysis
            'bos_bullish': bos_analysis['bullish'],
            'bos_bearish': bos_analysis['bearish'],
            'momentum_change': bos_analysis['momentum_change'],
            'fvgs': fvgs,
            
            # EMA alignment
            'bullish_alignment': current_price > current['ema20'] > current['ema50'] > current['ema200'],
            'bearish_alignment': current_price < current['ema20'] < current['ema50'] < current['ema200'],
            'price_above_ema20': current_price > current['ema20'],
            
            # Market regime
            'market_regime': self._determine_market_regime(df),
            'volatility_level': self._classify_volatility(atr_pct)
        }
    
    def _detect_enhanced_bos(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Enhanced Break of Structure detection"""
        try:
            # Simple BOS detection with multiple confirmations
            highs = df['high'].rolling(10).max()
            lows = df['low'].rolling(10).min()
            
            current_price = df['close'].iloc[-1]
            recent_high = highs.iloc[-5:].max()
            recent_low = lows.iloc[-5:].min()
            
            # Volume confirmation
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_confirmation = current_volume > avg_volume * 1.2
            
            # Momentum confirmation
            price_momentum = df['close'].pct_change(5).iloc[-1]
            strong_momentum = abs(price_momentum) > 0.02  # 2% move
            
            bos_bullish = (current_price > recent_high and 
                          volume_confirmation and 
                          price_momentum > 0)
            
            bos_bearish = (current_price < recent_low and 
                          volume_confirmation and 
                          price_momentum < 0)
            
            return {
                'bullish': bos_bullish,
                'bearish': bos_bearish,
                'momentum_change': strong_momentum
            }
        except Exception as e:
            print(f"Error in BOS detection: {e}")
            return {'bullish': False, 'bearish': False, 'momentum_change': False}
    
    def _identify_enhanced_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """Enhanced Fair Value Gap identification"""
        fvgs = []
        
        try:
            for i in range(2, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                before = df.iloc[i-2]
                
                # Enhanced FVG with volume confirmation
                vol_multiplier = current['volume'] / df['volume'].rolling(10).mean().iloc[i]
                
                # Bullish FVG with volume confirmation
                if (previous['high'] < current['low'] and vol_multiplier > 1.1):
                    fvg = {
                        'type': 'bullish',
                        'start_idx': i-1,
                        'end_idx': i,
                        'top': current['low'],
                        'bottom': previous['high'],
                        'timestamp': current.name,
                        'volume_multiplier': vol_multiplier,
                        'strength': 'strong' if vol_multiplier > 1.5 else 'moderate',
                        'filled': False
                    }
                    fvgs.append(fvg)
                
                # Bearish FVG with volume confirmation
                elif (previous['low'] > current['high'] and vol_multiplier > 1.1):
                    fvg = {
                        'type': 'bearish',
                        'start_idx': i-1,
                        'end_idx': i,
                        'top': previous['low'],
                        'bottom': current['high'],
                        'timestamp': current.name,
                        'volume_multiplier': vol_multiplier,
                        'strength': 'strong' if vol_multiplier > 1.5 else 'moderate',
                        'filled': False
                    }
                    fvgs.append(fvg)
        
        except Exception as e:
            print(f"Error identifying FVGs: {e}")
        
        return fvgs[-10:]  # Keep last 10 FVGs
    
    def _generate_enhanced_confluences(self, data: Dict[str, Any]) -> List[str]:
        """Generate enhanced confluence list with scoring"""
        confluences = []
        
        # BOS/CoC Confluence
        if data.get('bos_bullish'):
            confluences.append("✅ Bullish Break of Structure confirmed")
        elif data.get('bos_bearish'):
            confluences.append("✅ Bearish Break of Structure confirmed")
        elif data.get('momentum_change'):
            confluences.append("🔄 Momentum change detected")
        
        # FVG Confluence
        fvgs = data.get('fvgs', [])
        strong_fvgs = [fvg for fvg in fvgs if fvg.get('strength') == 'strong']
        if strong_fvgs:
            confluences.append("✅ Strong FVG zone entry")
        elif fvgs:
            confluences.append("⚡ FVG zone available")
        
        # EMA Confluence
        if data.get('bullish_alignment'):
            confluences.append("✅ Bullish EMA alignment (20>50>200)")
        elif data.get('bearish_alignment'):
            confluences.append("✅ Bearish EMA alignment (20<50<200)")
        elif data.get('price_above_ema20'):
            confluences.append("📈 Price above EMA 20")
        
        # RSI Confluence
        rsi = data.get('rsi', 50)
        if rsi > 70:
            confluences.append("🔴 RSI overbought (>70) - reversal zone")
        elif rsi < 30:
            confluences.append("🟢 RSI oversold (<30) - reversal zone")
        elif rsi > 60:
            confluences.append("📈 RSI bullish momentum (>60)")
        elif rsi < 40:
            confluences.append("📉 RSI bearish momentum (<40)")
        
        # Volume Confluence
        if data.get('volume_ratio', 1) > 1.5:
            confluences.append("✅ Strong volume confirmation (1.5x)")
        elif data.get('volume_ratio', 1) > 1.2:
            confluences.append("📊 Volume confirmation (1.2x)")
        
        # OBV Confluence
        if data.get('obv_score', 0) >= 5:
            confluences.append(f"✅ OBV trend alignment ({data.get('obv_trend')})")
        
        # VWAP Confluence
        if data.get('vwap_score', 0) >= 5:
            confluences.append(f"📊 VWAP position favorable ({data.get('vwap_position')})")
        
        # ATR/Volatility Confluence
        if data.get('atr_score', 0) >= 5:
            confluences.append("⚡ Optimal volatility level")
        
        # Stochastic RSI Confluence
        stoch_k = data.get('stoch_rsi_k', 50)
        stoch_d = data.get('stoch_rsi_d', 50)
        if stoch_k > 80 and stoch_d > 80:
            confluences.append("🔴 Stoch RSI overbought - reversal signal")
        elif stoch_k < 20 and stoch_d < 20:
            confluences.append("🟢 Stoch RSI oversold - bounce signal")
        
        # Williams %R Confluence
        williams_r = data.get('williams_r', -50)
        if williams_r > -20:
            confluences.append("🔴 Williams %R overbought")
        elif williams_r < -80:
            confluences.append("🟢 Williams %R oversold")
        
        # CMF Confluence
        cmf = data.get('cmf', 0)
        if cmf > 0.1:
            confluences.append("💪 Strong buying pressure (CMF)")
        elif cmf < -0.1:
            confluences.append("🐻 Strong selling pressure (CMF)")
        
        return confluences
    
    def _determine_market_regime(self, df: pd.DataFrame) -> str:
        """Determine current market regime"""
        try:
            # Calculate trend strength
            ema20 = df['close'].ewm(span=20).mean()
            ema50 = df['close'].ewm(span=50).mean()
            
            current_price = df['close'].iloc[-1]
            price_change_5d = ((current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100
            
            # Volatility assessment
            atr_pct = df['close'].pct_change().std() * 100
            
            if abs(price_change_5d) > 5 and atr_pct > 3:
                return "HIGH_VOLATILITY_TRENDING"
            elif abs(price_change_5d) > 3:
                return "TRENDING"
            elif atr_pct > 4:
                return "HIGH_VOLATILITY_SIDEWAYS"
            else:
                return "CONSOLIDATING"
                
        except Exception:
            return "UNKNOWN"
    
    def _classify_volatility(self, atr_pct: float) -> str:
        """Classify volatility level"""
        if atr_pct > 4:
            return "HIGH"
        elif atr_pct > 2:
            return "MEDIUM"
        else:
            return "LOW"

# Demo function
async def demo_enhanced_analysis():
    """Demo enhanced market analysis"""
    print("🔬 Enhanced Market Analysis Demo")
    print("=" * 40)
    
    analyzer = EnhancedMarketAnalyzer()
    
    # สร้าง mock exchange data
    import random
    periods = 100
    base_price = 65000
    
    # Generate sample data
    data = []
    price = base_price
    volume_base = 5000
    
    for i in range(periods):
        # Realistic price movement
        trend = 0.001 * np.sin(i / 10)  # Cyclical trend
        noise = random.gauss(0, 0.02)
        price = price * (1 + trend + noise)
        
        # Volume with spikes
        volume = volume_base + random.randint(-2000, 8000)
        if i % 20 == 0:  # Volume spikes
            volume *= 2
        
        data.append([
            datetime.utcnow().timestamp() * 1000 - (periods - i) * 3600000,
            price * 0.999,  # open
            price * 1.002,  # high
            price * 0.998,  # low
            price,          # close
            volume
        ])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Mock exchange class
    class MockExchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return data[-limit:] if limit else data
    
    mock_exchange = MockExchange()
    
    # Run enhanced analysis
    result = await analyzer.comprehensive_enhanced_analysis(mock_exchange, 'BTC/USDT', '1h')
    
    print(f"\n📊 Enhanced Analysis Results:")
    print("=" * 50)
    print(f"💰 Current Price: ${result['current_price']:,.2f}")
    print(f"📈 24h Change: {result['price_change_24h']:+.2f}%")
    print(f"🎯 Signal Strength: {result['signal_strength']:.1f}%")
    print(f"🏛️ Market Regime: {result['market_regime']}")
    print(f"⚡ Volatility: {result['volatility_level']}")
    
    print(f"\n📊 Technical Indicators:")
    print(f"   RSI: {result['rsi']:.1f}")
    print(f"   ATR: {result['atr_pct']:.2f}%")
    print(f"   OBV Trend: {result['obv_trend']}")
    print(f"   VWAP Position: {result['vwap_position']}")
    print(f"   Volume Ratio: {result['volume_ratio']:.1f}x")
    print(f"   Williams %R: {result['williams_r']:.1f}")
    print(f"   CMF: {result['cmf']:.3f}")
    print(f"   Stoch RSI K: {result['stoch_rsi_k']:.1f}")
    
    print(f"\n✅ Confluences Found ({result['confluence_count']}):")
    for confluence in result['confluences']:
        print(f"   {confluence}")
    
    print(f"\n🎯 Enhanced Features:")
    print("   ✅ OBV trend analysis")
    print("   ✅ ATR volatility classification")
    print("   ✅ VWAP position analysis")
    print("   ✅ Multi-indicator signal strength")
    print("   ✅ Enhanced FVG detection with volume")
    print("   ✅ Weighted confluence scoring")

if __name__ == "__main__":
    asyncio.run(demo_enhanced_analysis())