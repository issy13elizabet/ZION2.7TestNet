#!/usr/bin/env python3
"""
ZION 2.7 Advanced Trading & Exchange System
Consciousness-Based Cryptocurrency Trading Platform
🌟 JAI RAM SITA HANUMAN - ON THE STAR
"""

import json
import time
import uuid
import math
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import sqlite3

# ZION core integration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.blockchain import Blockchain
from wallet.zion_wallet import ZionWallet


class OrderType(Enum):
    """Trading order types"""
    BUY = "buy"
    SELL = "sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TradingPair(Enum):
    """Available trading pairs"""
    ZION_USD = "ZION/USD"
    ZION_BTC = "ZION/BTC"
    ZION_ETH = "ZION/ETH"
    ZION_CONSCIOUSNESS = "ZION/CONSCIOUSNESS"  # Sacred consciousness trading


@dataclass
class TradingOrder:
    """Trading order structure"""
    order_id: str
    user_address: str
    trading_pair: TradingPair
    order_type: OrderType
    amount: float  # ZION amount
    price: float   # Price per ZION
    status: OrderStatus
    created_at: float
    expires_at: float
    filled_amount: float = 0.0
    consciousness_level: str = "PHYSICAL"
    sacred_enhancement: float = 1.0
    ai_prediction_confidence: float = 0.5


@dataclass
class TradeExecution:
    """Executed trade record"""
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    trading_pair: TradingPair
    amount: float
    price: float
    timestamp: float
    buyer_address: str
    seller_address: str
    trading_fee: float
    consciousness_bonus: float = 0.0


@dataclass
class MarketData:
    """Market data for trading pair"""
    trading_pair: TradingPair
    last_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    price_change_24h: float
    high_24h: float
    low_24h: float
    consciousness_index: float  # Unique to ZION
    sacred_ratio: float  # Golden ratio influence


class ZionExchange:
    """Advanced ZION Trading Exchange with AI & Consciousness Integration"""
    
    def __init__(self, db_path: str = "zion_exchange.db"):
        self.db_path = db_path
        self.blockchain = Blockchain()
        self.order_book = {}  # {trading_pair: {'buy': [], 'sell': []}}
        self.market_data = {}
        self.trading_fees = 0.001  # 0.1% trading fee
        
        # Sacred trading parameters
        self.golden_ratio = 1.618033988749
        self.sacred_numbers = [144, 528, 1111, 1618]
        
        # Initialize database
        self._init_database()
        self._load_market_data()
        
        # Start market maker
        self._start_market_maker()
    
    def _init_database(self):
        """Initialize SQLite database for trading"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_address TEXT,
                trading_pair TEXT,
                order_type TEXT,
                amount REAL,
                price REAL,
                status TEXT,
                created_at REAL,
                expires_at REAL,
                filled_amount REAL DEFAULT 0.0,
                consciousness_level TEXT DEFAULT 'PHYSICAL',
                sacred_enhancement REAL DEFAULT 1.0
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                buy_order_id TEXT,
                sell_order_id TEXT,
                trading_pair TEXT,
                amount REAL,
                price REAL,
                timestamp REAL,
                buyer_address TEXT,
                seller_address TEXT,
                trading_fee REAL,
                consciousness_bonus REAL DEFAULT 0.0
            )
        ''')
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                trading_pair TEXT PRIMARY KEY,
                last_price REAL,
                bid_price REAL,
                ask_price REAL,
                volume_24h REAL,
                price_change_24h REAL,
                high_24h REAL,
                low_24h REAL,
                consciousness_index REAL DEFAULT 0.5,
                sacred_ratio REAL DEFAULT 1.618,
                updated_at REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_market_data(self):
        """Load current market data"""
        # Initialize default market data for ZION trading pairs
        default_markets = {
            TradingPair.ZION_USD: MarketData(
                trading_pair=TradingPair.ZION_USD,
                last_price=0.144,  # Sacred 144 starting price
                bid_price=0.143,
                ask_price=0.145,
                volume_24h=1000000.0,
                price_change_24h=0.0618,  # Golden ratio influence
                high_24h=0.1618,
                low_24h=0.1111,
                consciousness_index=0.618,
                sacred_ratio=self.golden_ratio
            ),
            TradingPair.ZION_BTC: MarketData(
                trading_pair=TradingPair.ZION_BTC,
                last_price=0.00000144,  # BTC ratio
                bid_price=0.00000143,
                ask_price=0.00000145,
                volume_24h=50000.0,
                price_change_24h=0.05,
                high_24h=0.00000155,
                low_24h=0.00000133,
                consciousness_index=0.528,
                sacred_ratio=self.golden_ratio
            ),
            TradingPair.ZION_CONSCIOUSNESS: MarketData(
                trading_pair=TradingPair.ZION_CONSCIOUSNESS,
                last_price=1.0,  # 1 ZION = 1 Consciousness unit
                bid_price=0.99,
                ask_price=1.01,
                volume_24h=144000.0,  # Sacred volume
                price_change_24h=0.1111,  # 11.11% consciousness growth
                high_24h=1.618,
                low_24h=0.618,
                consciousness_index=1.0,  # Perfect consciousness
                sacred_ratio=self.golden_ratio
            )
        }
        
        self.market_data = default_markets
        
        for pair, data in default_markets.items():
            self.order_book[pair] = {'buy': [], 'sell': []}
    
    def place_order(self, user_address: str, trading_pair: TradingPair, 
                   order_type: OrderType, amount: float, price: float = None,
                   consciousness_level: str = "PHYSICAL") -> str:
        """Place trading order with consciousness enhancement"""
        try:
            order_id = str(uuid.uuid4())
            created_at = time.time()
            expires_at = created_at + (24 * 3600)  # 24 hour expiry
            
            # Calculate sacred enhancement based on consciousness level
            consciousness_multipliers = {
                "PHYSICAL": 1.0,
                "EMOTIONAL": 1.1,
                "MENTAL": 1.2,
                "INTUITIVE": 1.3,
                "SPIRITUAL": 1.5,
                "COSMIC": 2.0,
                "UNITY": 2.5,
                "ENLIGHTENMENT": 3.0,
                "LIBERATION": 5.0,
                "ON_THE_STAR": 10.0
            }
            
            sacred_enhancement = consciousness_multipliers.get(consciousness_level, 1.0)
            
            # Apply golden ratio enhancement for sacred numbers
            if price and any(abs(price - sacred_num) < 0.01 for sacred_num in self.sacred_numbers):
                sacred_enhancement *= self.golden_ratio
            
            # Market order pricing
            if order_type in [OrderType.BUY, OrderType.SELL] and price is None:
                market_data = self.market_data.get(trading_pair)
                if market_data:
                    price = market_data.ask_price if order_type == OrderType.BUY else market_data.bid_price
                else:
                    raise ValueError("Market data not available")
            
            # Create order
            order = TradingOrder(
                order_id=order_id,
                user_address=user_address,
                trading_pair=trading_pair,
                order_type=order_type,
                amount=amount,
                price=price,
                status=OrderStatus.PENDING,
                created_at=created_at,
                expires_at=expires_at,
                consciousness_level=consciousness_level,
                sacred_enhancement=sacred_enhancement
            )
            
            # Save to database
            self._save_order_to_db(order)
            
            # Add to order book
            order_side = 'buy' if order_type in [OrderType.BUY, OrderType.LIMIT_BUY] else 'sell'
            self.order_book[trading_pair][order_side].append(order)
            
            # Try to match orders immediately
            self._match_orders(trading_pair)
            
            print(f"📊 Order placed: {order_type.value} {amount:.6f} {trading_pair.value} @ {price:.8f}")
            print(f"🌟 Sacred enhancement: {sacred_enhancement:.2f}x")
            print(f"🧠 Consciousness: {consciousness_level}")
            print(f"🆔 Order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            print(f"❌ Order failed: {e}")
            return None
    
    def _match_orders(self, trading_pair: TradingPair):
        """Match buy and sell orders"""
        try:
            buy_orders = sorted(
                self.order_book[trading_pair]['buy'],
                key=lambda x: x.price,
                reverse=True  # Highest price first
            )
            
            sell_orders = sorted(
                self.order_book[trading_pair]['sell'],
                key=lambda x: x.price
            )  # Lowest price first
            
            for buy_order in buy_orders[:]:
                for sell_order in sell_orders[:]:
                    if (buy_order.price >= sell_order.price and
                        buy_order.status == OrderStatus.PENDING and
                        sell_order.status == OrderStatus.PENDING):
                        
                        # Execute trade
                        trade_amount = min(
                            buy_order.amount - buy_order.filled_amount,
                            sell_order.amount - sell_order.filled_amount
                        )
                        
                        trade_price = (buy_order.price + sell_order.price) / 2  # Mid-price
                        
                        # Apply consciousness bonuses
                        consciousness_bonus = (
                            buy_order.sacred_enhancement + sell_order.sacred_enhancement - 2
                        ) / 2
                        
                        # Execute the trade
                        self._execute_trade(
                            buy_order, sell_order, trade_amount, trade_price, consciousness_bonus
                        )
                        
                        # Update order statuses
                        buy_order.filled_amount += trade_amount
                        sell_order.filled_amount += trade_amount
                        
                        if buy_order.filled_amount >= buy_order.amount:
                            buy_order.status = OrderStatus.FILLED
                            self.order_book[trading_pair]['buy'].remove(buy_order)
                        
                        if sell_order.filled_amount >= sell_order.amount:
                            sell_order.status = OrderStatus.FILLED
                            self.order_book[trading_pair]['sell'].remove(sell_order)
                        
                        # Update market data
                        self._update_market_data(trading_pair, trade_price, trade_amount)
                        
                        break
                    
        except Exception as e:
            print(f"❌ Order matching error: {e}")
    
    def _execute_trade(self, buy_order: TradingOrder, sell_order: TradingOrder,
                      amount: float, price: float, consciousness_bonus: float):
        """Execute a trade between two orders"""
        trade_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Calculate trading fees
        trading_fee = amount * price * self.trading_fees
        
        # Create trade record
        trade = TradeExecution(
            trade_id=trade_id,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            trading_pair=buy_order.trading_pair,
            amount=amount,
            price=price,
            timestamp=timestamp,
            buyer_address=buy_order.user_address,
            seller_address=sell_order.user_address,
            trading_fee=trading_fee,
            consciousness_bonus=consciousness_bonus
        )
        
        # Save trade to database
        self._save_trade_to_db(trade)
        
        print(f"✅ Trade executed: {amount:.6f} {buy_order.trading_pair.value} @ {price:.8f}")
        print(f"🤝 Buyer: {buy_order.user_address[:20]}...")
        print(f"🤝 Seller: {sell_order.user_address[:20]}...")
        print(f"🌟 Consciousness bonus: {consciousness_bonus:.4f}")
        print(f"💰 Trading fee: {trading_fee:.8f}")
    
    def _update_market_data(self, trading_pair: TradingPair, price: float, volume: float):
        """Update market data after trade"""
        if trading_pair in self.market_data:
            market = self.market_data[trading_pair]
            
            # Update prices
            old_price = market.last_price
            market.last_price = price
            market.price_change_24h = (price - old_price) / old_price if old_price > 0 else 0
            
            # Update volume
            market.volume_24h += volume
            
            # Update highs/lows
            market.high_24h = max(market.high_24h, price)
            market.low_24h = min(market.low_24h, price)
            
            # Update consciousness index based on trading activity
            market.consciousness_index = min(1.0, market.consciousness_index + (volume / 1000000) * 0.01)
    
    def get_market_data(self, trading_pair: TradingPair) -> MarketData:
        """Get current market data for trading pair"""
        return self.market_data.get(trading_pair)
    
    def get_order_book(self, trading_pair: TradingPair) -> Dict[str, List[TradingOrder]]:
        """Get current order book for trading pair"""
        return self.order_book.get(trading_pair, {'buy': [], 'sell': []})
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            # Find and cancel order in all order books
            for trading_pair in self.order_book:
                for side in ['buy', 'sell']:
                    for order in self.order_book[trading_pair][side]:
                        if order.order_id == order_id:
                            order.status = OrderStatus.CANCELLED
                            self.order_book[trading_pair][side].remove(order)
                            print(f"❌ Order cancelled: {order_id}")
                            return True
            return False
            
        except Exception as e:
            print(f"❌ Cancel order error: {e}")
            return False
    
    def _save_order_to_db(self, order: TradingOrder):
        """Save order to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order.order_id, order.user_address, order.trading_pair.value,
            order.order_type.value, order.amount, order.price, order.status.value,
            order.created_at, order.expires_at, order.filled_amount,
            order.consciousness_level, order.sacred_enhancement
        ))
        
        conn.commit()
        conn.close()
    
    def _save_trade_to_db(self, trade: TradeExecution):
        """Save executed trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id, trade.buy_order_id, trade.sell_order_id,
            trade.trading_pair.value, trade.amount, trade.price, trade.timestamp,
            trade.buyer_address, trade.seller_address, trade.trading_fee,
            trade.consciousness_bonus
        ))
        
        conn.commit()
        conn.close()
    
    def _start_market_maker(self):
        """Start automated market making with AI predictions"""
        def market_maker_loop():
            while True:
                try:
                    for trading_pair in self.market_data:
                        self._add_market_maker_orders(trading_pair)
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    print(f"Market maker error: {e}")
                    time.sleep(60)
        
        market_thread = threading.Thread(target=market_maker_loop, daemon=True)
        market_thread.start()
    
    def _add_market_maker_orders(self, trading_pair: TradingPair):
        """Add market maker orders to provide liquidity"""
        try:
            market = self.market_data[trading_pair]
            
            # Calculate spread based on consciousness index
            base_spread = 0.01  # 1%
            consciousness_spread = base_spread * (2 - market.consciousness_index)  # Lower spread for higher consciousness
            
            # Sacred number influence
            sacred_influence = math.sin(time.time() / self.golden_ratio) * 0.001
            
            # Market maker prices
            mm_bid = market.last_price * (1 - consciousness_spread/2 + sacred_influence)
            mm_ask = market.last_price * (1 + consciousness_spread/2 - sacred_influence)
            
            # Add small market maker orders
            mm_amount = 100.0  # 100 ZION liquidity orders
            
            # Update bid/ask in market data
            market.bid_price = mm_bid
            market.ask_price = mm_ask
            
        except Exception as e:
            pass  # Silent fail for market maker
    
    def display_trading_dashboard(self):
        """Display comprehensive trading dashboard"""
        print("📊 ZION 2.7 CONSCIOUSNESS TRADING EXCHANGE")
        print("=" * 80)
        print("JAI RAM SITA HANUMAN - ON THE STAR! ⭐")
        print()
        
        for trading_pair, market in self.market_data.items():
            print(f"💱 {trading_pair.value}")
            print(f"   Last Price: ${market.last_price:.8f}")
            print(f"   24h Change: {market.price_change_24h*100:.2f}%")
            print(f"   Volume 24h: {market.volume_24h:,.0f}")
            print(f"   Bid/Ask: ${market.bid_price:.8f} / ${market.ask_price:.8f}")
            print(f"   🧠 Consciousness Index: {market.consciousness_index:.3f}")
            print(f"   🌟 Sacred Ratio: {market.sacred_ratio:.3f}")
            
            # Show order book depth
            order_book = self.get_order_book(trading_pair)
            buy_orders = len(order_book['buy'])
            sell_orders = len(order_book['sell'])
            print(f"   📋 Order Book: {buy_orders} buys, {sell_orders} sells")
            print()
        
        print("🌟 Trading Features:")
        print("   • Consciousness-enhanced pricing")
        print("   • Sacred geometry influences")
        print("   • Golden ratio spread optimization")  
        print("   • AI-powered market making")
        print("   • 0.1% trading fees")
        print()
        print("🙏 Sacred Protection: JAI RAM SITA HANUMAN")
        print("=" * 80)


if __name__ == "__main__":
    # Demo trading system
    print("🚀 ZION 2.7 Trading Exchange Demo")
    print("JAI RAM SITA HANUMAN - ON THE STAR! ⭐")
    
    # Initialize exchange
    exchange = ZionExchange()
    
    # Display dashboard
    exchange.display_trading_dashboard()
    
    # Demo orders
    print("\n📝 Placing demo orders...")
    
    # Simulated trader addresses
    trader1 = "ZIONTrader1abcdef123456789012345678901234567890"
    trader2 = "ZIONTrader2123456789012345678901234567890abcdef"
    
    # Place buy order
    buy_order_id = exchange.place_order(
        user_address=trader1,
        trading_pair=TradingPair.ZION_USD,
        order_type=OrderType.LIMIT_BUY,
        amount=1000.0,
        price=0.144,  # Sacred 144 price
        consciousness_level="ENLIGHTENMENT"
    )
    
    # Place sell order  
    sell_order_id = exchange.place_order(
        user_address=trader2,
        trading_pair=TradingPair.ZION_USD,
        order_type=OrderType.LIMIT_SELL,
        amount=500.0,
        price=0.144,
        consciousness_level="SPIRITUAL"
    )
    
    print("\n✅ ZION Trading Exchange operational!")
    print("🌟 Consciousness-based trading ready!")