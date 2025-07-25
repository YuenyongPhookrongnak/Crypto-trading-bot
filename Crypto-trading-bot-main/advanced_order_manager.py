#!/usr/bin/env python3
"""
Advanced Order Execution System

ระบบ execution ที่ทันสมัย:
- Market Order สำหรับ urgent entries
- Limit Order สำหรับ better fills
- Trailing Stop Loss (ตาม BOS)
- Smart order routing
- Slippage protection
"""

import asyncio
import ccxt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP = "TRAILING_STOP"

class OrderUrgency(Enum):
    IMMEDIATE = "IMMEDIATE"  # Market order
    NORMAL = "NORMAL"       # Limit order with small spread
    PATIENT = "PATIENT"     # Limit order with better price

class TrailingStopType(Enum):
    PERCENTAGE = "PERCENTAGE"
    BOS_BASED = "BOS_BASED"     # Follow Break of Structure
    ATR_BASED = "ATR_BASED"     # Based on ATR distance

@dataclass
class OrderRequest:
    """Order request with advanced parameters"""
    symbol: str
    side: str  # buy/sell
    amount: float
    order_type: OrderType
    urgency: OrderUrgency
    
    # Price parameters
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trigger_price: Optional[float] = None
    
    # Advanced features
    max_slippage_pct: float = 0.5  # 0.5% max slippage
    time_in_force: str = "GTC"     # GTC, IOC, FOK
    reduce_only: bool = False
    
    # Trailing stop parameters
    trailing_type: Optional[TrailingStopType] = None
    trailing_distance: Optional[float] = None
    
    # Execution strategy
    split_order: bool = False
    split_parts: int = 1
    execution_delay: float = 0  # seconds between parts

@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    status: str  # filled, partial, cancelled, pending
    filled_amount: float
    average_price: float
    fees: float
    slippage_pct: float
    execution_time: float
    error: Optional[str] = None

class SmartOrderRouter:
    """Smart order routing system"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.trailing_stops = {}  # Track active trailing stops
        self.order_history = []
        
    async def execute_smart_order(self, request: OrderRequest) -> OrderResult:
        """Execute order with smart routing"""
        try:
            start_time = time.time()
            
            # Choose execution strategy based on urgency and market conditions
            if request.urgency == OrderUrgency.IMMEDIATE:
                result = await self._execute_market_order(request)
            elif request.urgency == OrderUrgency.NORMAL:
                result = await self._execute_smart_limit_order(request)
            else:  # PATIENT
                result = await self._execute_patient_limit_order(request)
            
            result.execution_time = time.time() - start_time
            self.order_history.append(result)
            
            return result
            
        except Exception as e:
            return OrderResult(
                order_id="",
                status="error",
                filled_amount=0,
                average_price=0,
                fees=0,
                slippage_pct=0,
                execution_time=0,
                error=str(e)
            )
    
    async def _execute_market_order(self, request: OrderRequest) -> OrderResult:
        """Execute immediate market order with slippage protection"""
        try:
            # Get current market price
            ticker = self.exchange.fetch_ticker(request.symbol)
            current_price = ticker['bid'] if request.side == 'sell' else ticker['ask']
            
            # Check spread for slippage estimation
            spread_pct = (ticker['ask'] - ticker['bid']) / ticker['last'] * 100
            
            if spread_pct > request.max_slippage_pct:
                print(f"⚠️ High spread detected: {spread_pct:.2f}% > {request.max_slippage_pct}%")
                # Fall back to limit order
                request.limit_price = current_price
                return await self._execute_smart_limit_order(request)
            
            # Execute market order
            if request.split_order and request.split_parts > 1:
                return await self._execute_split_market_order(request)
            else:
                order = self.exchange.create_market_order(
                    request.symbol, 
                    request.side, 
                    request.amount,
                    None,
                    None,
                    {'timeInForce': request.time_in_force}
                )
                
                # Calculate slippage
                executed_price = float(order.get('average', current_price))
                slippage = abs(executed_price - current_price) / current_price * 100
                
                return OrderResult(
                    order_id=order['id'],
                    status=order['status'],
                    filled_amount=float(order.get('filled', 0)),
                    average_price=executed_price,
                    fees=float(order.get('fee', {}).get('cost', 0)),
                    slippage_pct=slippage,
                    execution_time=0
                )
                
        except Exception as e:
            print(f"❌ Market order failed: {e}")
            raise
    
    async def _execute_smart_limit_order(self, request: OrderRequest) -> OrderResult:
        """Execute smart limit order with optimal pricing"""
        try:
            # Get order book for better pricing
            orderbook = self.exchange.fetch_order_book(request.symbol)
            
            # Calculate optimal limit price
            if not request.limit_price:
                request.limit_price = self._calculate_smart_limit_price(
                    request, orderbook
                )
            
            # Place limit order
            order = self.exchange.create_limit_order(
                request.symbol,
                request.side,
                request.amount,
                request.limit_price,
                None,
                {'timeInForce': request.time_in_force}
            )
            
            # Monitor order for partial fills
            result = await self._monitor_limit_order(order, request)
            
            return result
            
        except Exception as e:
            print(f"❌ Smart limit order failed: {e}")
            raise
    
    async def _execute_patient_limit_order(self, request: OrderRequest) -> OrderResult:
        """Execute patient limit order with better pricing"""
        try:
            # Use more aggressive pricing for better fills
            orderbook = self.exchange.fetch_order_book(request.symbol)
            
            if request.side == 'buy':
                # Place bid slightly above best bid
                best_bid = orderbook['bids'][0][0] if orderbook['bids'] else orderbook['asks'][0][0] * 0.999
                request.limit_price = best_bid + (best_bid * 0.001)  # 0.1% above best bid
            else:
                # Place ask slightly below best ask
                best_ask = orderbook['asks'][0][0] if orderbook['asks'] else orderbook['bids'][0][0] * 1.001
                request.limit_price = best_ask - (best_ask * 0.001)  # 0.1% below best ask
            
            return await self._execute_smart_limit_order(request)
            
        except Exception as e:
            print(f"❌ Patient limit order failed: {e}")
            raise
    
    def _calculate_smart_limit_price(self, request: OrderRequest, orderbook: Dict) -> float:
        """Calculate optimal limit price based on order book"""
        try:
            if request.side == 'buy':
                # For buy orders, use best ask or slightly better
                if orderbook['asks']:
                    best_ask = orderbook['asks'][0][0]
                    if request.urgency == OrderUrgency.NORMAL:
                        return best_ask  # Take the spread
                    else:
                        return best_ask * 0.9995  # Slightly better price
                else:
                    return orderbook['bids'][0][0] if orderbook['bids'] else 0
            else:
                # For sell orders, use best bid or slightly better
                if orderbook['bids']:
                    best_bid = orderbook['bids'][0][0]
                    if request.urgency == OrderUrgency.NORMAL:
                        return best_bid  # Take the spread
                    else:
                        return best_bid * 1.0005  # Slightly better price
                else:
                    return orderbook['asks'][0][0] if orderbook['asks'] else 0
                    
        except Exception as e:
            print(f"❌ Error calculating limit price: {e}")
            # Fallback to current price
            ticker = self.exchange.fetch_ticker(request.symbol)
            return ticker['last']
    
    async def _monitor_limit_order(self, order: Dict, request: OrderRequest, 
                                 timeout: int = 300) -> OrderResult:
        """Monitor limit order execution"""
        start_time = time.time()
        order_id = order['id']
        
        while time.time() - start_time < timeout:
            try:
                # Check order status
                updated_order = self.exchange.fetch_order(order_id, request.symbol)
                
                if updated_order['status'] in ['closed', 'filled']:
                    # Order completely filled
                    return OrderResult(
                        order_id=order_id,
                        status='filled',
                        filled_amount=float(updated_order.get('filled', 0)),
                        average_price=float(updated_order.get('average', 0)),
                        fees=float(updated_order.get('fee', {}).get('cost', 0)),
                        slippage_pct=0,  # No slippage for limit orders
                        execution_time=time.time() - start_time
                    )
                
                elif updated_order['status'] == 'canceled':
                    return OrderResult(
                        order_id=order_id,
                        status='cancelled',
                        filled_amount=float(updated_order.get('filled', 0)),
                        average_price=float(updated_order.get('average', 0)),
                        fees=float(updated_order.get('fee', {}).get('cost', 0)),
                        slippage_pct=0,
                        execution_time=time.time() - start_time
                    )
                
                # Wait before checking again
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"⚠️ Error monitoring order: {e}")
                await asyncio.sleep(10)
        
        # Timeout reached - return partial fill status
        try:
            final_order = self.exchange.fetch_order(order_id, request.symbol)
            return OrderResult(
                order_id=order_id,
                status='partial',
                filled_amount=float(final_order.get('filled', 0)),
                average_price=float(final_order.get('average', 0)),
                fees=float(final_order.get('fee', {}).get('cost', 0)),
                slippage_pct=0,
                execution_time=timeout
            )
        except:
            return OrderResult(
                order_id=order_id,
                status='timeout',
                filled_amount=0,
                average_price=0,
                fees=0,
                slippage_pct=0,
                execution_time=timeout
            )
    
    async def _execute_split_market_order(self, request: OrderRequest) -> OrderResult:
        """Execute split market order to reduce market impact"""
        try:
            total_filled = 0
            total_fees = 0
            weighted_price = 0
            part_size = request.amount / request.split_parts
            
            for i in range(request.split_parts):
                # Execute part
                part_order = self.exchange.create_market_order(
                    request.symbol,
                    request.side,
                    part_size
                )
                
                filled = float(part_order.get('filled', 0))
                price = float(part_order.get('average', 0))
                fees = float(part_order.get('fee', {}).get('cost', 0))
                
                total_filled += filled
                total_fees += fees
                weighted_price += price * filled
                
                # Wait between parts
                if i < request.split_parts - 1 and request.execution_delay > 0:
                    await asyncio.sleep(request.execution_delay)
            
            avg_price = weighted_price / total_filled if total_filled > 0 else 0
            
            return OrderResult(
                order_id=f"SPLIT_{int(time.time())}",
                status='filled',
                filled_amount=total_filled,
                average_price=avg_price,
                fees=total_fees,
                slippage_pct=0,  # Calculate separately for split orders
                execution_time=0
            )
            
        except Exception as e:
            print(f"❌ Split order execution failed: {e}")
            raise

class TrailingStopManager:
    """Advanced trailing stop management"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.active_trails = {}
        self.market_data_cache = {}
        
    async def create_trailing_stop(self, symbol: str, side: str, amount: float,
                                 trail_type: TrailingStopType, trail_distance: float,
                                 initial_price: float) -> str:
        """Create trailing stop order"""
        trail_id = f"TRAIL_{symbol}_{int(time.time())}"
        
        self.active_trails[trail_id] = {
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'trail_type': trail_type,
            'trail_distance': trail_distance,
            'initial_price': initial_price,
            'best_price': initial_price,
            'stop_price': self._calculate_initial_stop(initial_price, side, trail_distance, trail_type),
            'created_at': datetime.utcnow(),
            'status': 'active'
        }
        
        print(f"✅ Trailing stop created: {trail_id}")
        print(f"   Symbol: {symbol} {side}")
        print(f"   Initial price: ${initial_price:.2f}")
        print(f"   Stop price: ${self.active_trails[trail_id]['stop_price']:.2f}")
        
        return trail_id
    
    def _calculate_initial_stop(self, price: float, side: str, distance: float, 
                              trail_type: TrailingStopType) -> float:
        """Calculate initial stop price"""
        if trail_type == TrailingStopType.PERCENTAGE:
            if side == 'buy':  # Long position
                return price * (1 - distance / 100)
            else:  # Short position
                return price * (1 + distance / 100)
        elif trail_type == TrailingStopType.ATR_BASED:
            # Use ATR as distance (simplified)
            atr_value = distance  # distance should be ATR value
            if side == 'buy':
                return price - atr_value
            else:
                return price + atr_value
        else:  # BOS_BASED
            # Use structure levels (simplified)
            if side == 'buy':
                return price * (1 - distance / 100)  # Fallback to percentage
            else:
                return price * (1 + distance / 100)
    
    async def update_trailing_stops(self):
        """Update all active trailing stops"""
        for trail_id, trail_data in list(self.active_trails.items()):
            if trail_data['status'] != 'active':
                continue
                
            try:
                await self._update_single_trail(trail_id, trail_data)
            except Exception as e:
                print(f"❌ Error updating trail {trail_id}: {e}")
    
    async def _update_single_trail(self, trail_id: str, trail_data: Dict):
        """Update single trailing stop"""
        symbol = trail_data['symbol']
        side = trail_data['side']
        
        # Get current price
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Check if we need to update the trailing stop
        updated = False
        
        if side == 'buy':  # Long position - trail up
            if current_price > trail_data['best_price']:
                trail_data['best_price'] = current_price
                new_stop = self._calculate_new_stop_price(trail_data, current_price)
                
                if new_stop > trail_data['stop_price']:
                    trail_data['stop_price'] = new_stop
                    updated = True
                    
        else:  # Short position - trail down
            if current_price < trail_data['best_price']:
                trail_data['best_price'] = current_price
                new_stop = self._calculate_new_stop_price(trail_data, current_price)
                
                if new_stop < trail_data['stop_price']:
                    trail_data['stop_price'] = new_stop
                    updated = True
        
        # Check if stop should trigger
        stop_triggered = self._check_stop_trigger(trail_data, current_price)
        
        if stop_triggered:
            await self._execute_trailing_stop(trail_id, trail_data, current_price)
        elif updated:
            print(f"📈 Trailing stop updated: {trail_id}")
            print(f"   New stop: ${trail_data['stop_price']:.2f}")
            print(f"   Current: ${current_price:.2f}")
    
    def _calculate_new_stop_price(self, trail_data: Dict, current_price: float) -> float:
        """Calculate new stop price for trailing stop"""
        trail_type = trail_data['trail_type']
        distance = trail_data['trail_distance']
        side = trail_data['side']
        
        if trail_type == TrailingStopType.PERCENTAGE:
            if side == 'buy':
                return current_price * (1 - distance / 100)
            else:
                return current_price * (1 + distance / 100)
        elif trail_type == TrailingStopType.ATR_BASED:
            if side == 'buy':
                return current_price - distance
            else:
                return current_price + distance
        else:  # BOS_BASED
            # Simplified BOS trailing (would need actual structure analysis)
            if side == 'buy':
                return current_price * (1 - distance / 100)
            else:
                return current_price * (1 + distance / 100)
    
    def _check_stop_trigger(self, trail_data: Dict, current_price: float) -> bool:
        """Check if trailing stop should trigger"""
        stop_price = trail_data['stop_price']
        side = trail_data['side']
        
        if side == 'buy':  # Long position
            return current_price <= stop_price
        else:  # Short position
            return current_price >= stop_price
    
    async def _execute_trailing_stop(self, trail_id: str, trail_data: Dict, current_price: float):
        """Execute trailing stop as market order"""
        try:
            symbol = trail_data['symbol']
            side = 'sell' if trail_data['side'] == 'buy' else 'buy'  # Opposite side to close
            amount = trail_data['amount']
            
            # Execute market order to close position
            order = self.exchange.create_market_order(symbol, side, amount)
            
            # Update trail status
            trail_data['status'] = 'executed'
            trail_data['executed_at'] = datetime.utcnow()
            trail_data['execution_price'] = current_price
            trail_data['order_id'] = order['id']
            
            print(f"🎯 Trailing stop executed: {trail_id}")
            print(f"   Execution price: ${current_price:.2f}")
            print(f"   Order ID: {order['id']}")
            
        except Exception as e:
            print(f"❌ Failed to execute trailing stop {trail_id}: {e}")
            trail_data['status'] = 'error'
            trail_data['error'] = str(e)

class AdvancedOrderManager:
    """Complete advanced order management system"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.order_router = SmartOrderRouter(exchange)
        self.trailing_manager = TrailingStopManager(exchange)
        
        # Start trailing stop monitor
        asyncio.create_task(self._trailing_stop_monitor())
    
    async def _trailing_stop_monitor(self):
        """Background task to monitor trailing stops"""
        while True:
            try:
                await self.trailing_manager.update_trailing_stops()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"❌ Trailing stop monitor error: {e}")
                await asyncio.sleep(60)
    
    async def place_entry_order(self, symbol: str, side: str, amount: float,
                              market_condition: str, urgency: str = "normal") -> OrderResult:
        """Place optimized entry order"""
        
        # Determine order type based on conditions
        if urgency == "immediate" or market_condition == "breakout":
            order_urgency = OrderUrgency.IMMEDIATE
        elif market_condition == "volatile":
            order_urgency = OrderUrgency.NORMAL
        else:
            order_urgency = OrderUrgency.PATIENT
        
        request = OrderRequest(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type=OrderType.MARKET if order_urgency == OrderUrgency.IMMEDIATE else OrderType.LIMIT,
            urgency=order_urgency,
            max_slippage_pct=0.5
        )
        
        return await self.order_router.execute_smart_order(request)
    
    async def place_stop_loss(self, symbol: str, side: str, amount: float,
                            stop_price: float, trailing: bool = False,
                            trail_distance: float = 2.0) -> str:
        """Place stop loss order with optional trailing"""
        
        if trailing:
            # Create trailing stop
            current_price = self.exchange.fetch_ticker(symbol)['last']
            return await self.trailing_manager.create_trailing_stop(
                symbol, side, amount, TrailingStopType.PERCENTAGE, 
                trail_distance, current_price
            )
        else:
            # Regular stop loss
            close_side = 'sell' if side == 'buy' else 'buy'
            order = self.exchange.create_order(
                symbol, OrderType.STOP_MARKET.value, close_side, amount,
                None, None, {'stopPrice': stop_price}
            )
            return order['id']
    
    async def place_take_profit_levels(self, symbol: str, side: str, 
                                     total_amount: float, tp_levels: List[float],
                                     partial_amounts: List[float]) -> List[str]:
        """Place multiple take profit levels"""
        order_ids = []
        close_side = 'sell' if side == 'buy' else 'buy'
        
        for tp_price, amount in zip(tp_levels, partial_amounts):
            try:
                order = self.exchange.create_order(
                    symbol, OrderType.TAKE_PROFIT_MARKET.value, close_side, amount,
                    None, None, {'stopPrice': tp_price}
                )
                order_ids.append(order['id'])
                print(f"💰 TP level set: ${tp_price:.2f} for {amount:.6f}")
            except Exception as e:
                print(f"❌ Failed to set TP level ${tp_price:.2f}: {e}")
        
        return order_ids

# Demo function
async def demo_advanced_execution():
    """Demo advanced order execution"""
    print("⚡ Advanced Order Execution Demo")
    print("=" * 40)
    
    # Mock exchange for demo
    class MockExchange:
        def fetch_ticker(self, symbol):
            return {'last': 50000, 'bid': 49995, 'ask': 50005}
        
        def fetch_order_book(self, symbol):
            return {
                'bids': [[49995, 1.5], [49990, 2.0]],
                'asks': [[50005, 1.2], [50010, 1.8]]
            }
        
        def create_market_order(self, symbol, side, amount, price=None, params=None):
            return {
                'id': f'ORDER_{int(time.time())}',
                'status': 'filled',
                'filled': amount,
                'average': 50000,
                'fee': {'cost': 0.1}
            }
        
        def create_limit_order(self, symbol, side, amount, price, params=None):
            return {
                'id': f'LIMIT_{int(time.time())}',
                'status': 'open',
                'filled': 0,
                'average': 0,
                'fee': {'cost': 0}
            }
        
        def fetch_order(self, order_id, symbol):
            return {
                'id': order_id,
                'status': 'filled',
                'filled': 0.001,
                'average': 50000,
                'fee': {'cost': 0.05}
            }
        
        def create_order(self, symbol, order_type, side, amount, price=None, params=None):
            return {
                'id': f'STOP_{int(time.time())}',
                'status': 'open'
            }
    
    mock_exchange = MockExchange()
    order_manager = AdvancedOrderManager(mock_exchange)
    
    print("🎯 Testing different order scenarios:")
    
    # Scenario 1: Immediate breakout entry
    print("\n1. Immediate Breakout Entry (Market Order)")
    result = await order_manager.place_entry_order(
        'BTC/USDT', 'buy', 0.001, 'breakout', 'immediate'
    )
    print(f"   Status: {result.status}")
    print(f"   Price: ${result.average_price:,.2f}")
    print(f"   Slippage: {result.slippage_pct:.2f}%")
    
    # Scenario 2: Patient consolidation entry
    print("\n2. Patient Consolidation Entry (Limit Order)")
    result = await order_manager.place_entry_order(
        'BTC/USDT', 'buy', 0.001, 'consolidation', 'patient'
    )
    print(f"   Status: {result.status}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    
    # Scenario 3: Trailing stop loss
    print("\n3. Trailing Stop Loss")
    trail_id = await order_manager.place_stop_loss(
        'BTC/USDT', 'buy', 0.001, 49000, trailing=True, trail_distance=2.0
    )
    print(f"   Trailing stop ID: {trail_id}")
    
    # Scenario 4: Multiple take profit levels
    print("\n4. Multiple Take Profit Levels")
    tp_orders = await order_manager.place_take_profit_levels(
        'BTC/USDT', 'buy', 0.001,
        [51000, 52000, 53000],  # TP prices
        [0.0004, 0.0003, 0.0003]  # Partial amounts
    )
    print(f"   TP orders placed: {len(tp_orders)}")
    
    print("\n✅ Advanced execution demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_advanced_execution())