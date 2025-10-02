"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import Link from "next/link";

interface TradingPair {
  symbol: string;
  base: string;
  quote: string;
  price: number;
  change_24h: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
  cosmic_alignment: number;
}

interface Order {
  id: string;
  pair: string;
  type: 'buy' | 'sell';
  order_type: 'market' | 'limit';
  amount: number;
  price: number;
  filled: number;
  status: 'pending' | 'filled' | 'cancelled';
  timestamp: string;
  dharma_impact: number;
}

interface Portfolio {
  asset: string;
  balance: number;
  value_usd: number;
  change_24h: number;
}

export default function TradingPage() {
  const [tradingPairs, setTradingPairs] = useState<TradingPair[]>([
    {
      symbol: 'ZION/USDT',
      base: 'ZION',
      quote: 'USDT',
      price: 42.108,
      change_24h: 21.69,
      volume_24h: 1337420,
      high_24h: 43.21,
      low_24h: 40.88,
      cosmic_alignment: 96
    },
    {
      symbol: 'ZION/BTC',
      base: 'ZION',
      quote: 'BTC',
      price: 0.000621,
      change_24h: -3.42,
      volume_24h: 88216,
      high_24h: 0.000644,
      low_24h: 0.000601,
      cosmic_alignment: 88
    },
    {
      symbol: 'DHARMA/ZION',
      base: 'DHARMA',
      quote: 'ZION',
      price: 2.108,
      change_24h: 108.42,
      volume_24h: 216432,
      high_24h: 2.888,
      low_24h: 1.069,
      cosmic_alignment: 100
    }
  ]);

  const [orders, setOrders] = useState<Order[]>([
    {
      id: 'ord_001',
      pair: 'ZION/USDT',
      type: 'buy',
      order_type: 'limit',
      amount: 10,
      price: 41.5,
      filled: 0,
      status: 'pending',
      timestamp: '2024-09-25 14:30:00',
      dharma_impact: 5
    },
    {
      id: 'ord_002',
      pair: 'DHARMA/ZION',
      type: 'sell',
      order_type: 'market',
      amount: 50,
      price: 2.08,
      filled: 50,
      status: 'filled',
      timestamp: '2024-09-25 13:15:00',
      dharma_impact: -2
    }
  ]);

  const [portfolio, setPortfolio] = useState<Portfolio[]>([
    {
      asset: 'ZION',
      balance: 1337.42,
      value_usd: 56342.18,
      change_24h: 21.69
    },
    {
      asset: 'DHARMA',
      balance: 108.21,
      value_usd: 9563.44,
      change_24h: 108.42
    },
    {
      asset: 'BTC',
      balance: 0.21084,
      value_usd: 14216.33,
      change_24h: 2.16
    },
    {
      asset: 'USDT',
      balance: 4269.21,
      value_usd: 4269.21,
      change_24h: 0
    }
  ]);

  const [selectedPair, setSelectedPair] = useState<TradingPair | null>(tradingPairs[0]);
  const [orderForm, setOrderForm] = useState({
    type: 'buy' as 'buy' | 'sell',
    order_type: 'limit' as 'market' | 'limit',
    amount: '',
    price: ''
  });

  const handlePlaceOrder = async () => {
    if (!selectedPair || !orderForm.amount || (orderForm.order_type === 'limit' && !orderForm.price)) {
      alert('Please fill in all order details');
      return;
    }

    const newOrder: Order = {
      id: `ord_${Date.now()}`,
      pair: selectedPair.symbol,
      type: orderForm.type,
      order_type: orderForm.order_type,
      amount: parseFloat(orderForm.amount),
      price: orderForm.order_type === 'market' ? selectedPair.price : parseFloat(orderForm.price),
      filled: orderForm.order_type === 'market' ? parseFloat(orderForm.amount) : 0,
      status: orderForm.order_type === 'market' ? 'filled' : 'pending',
      timestamp: new Date().toLocaleString(),
      dharma_impact: Math.floor(Math.random() * 10) - 5
    };

    setOrders([newOrder, ...orders]);
    setOrderForm({ type: 'buy', order_type: 'limit', amount: '', price: '' });
  };

  const cancelOrder = (orderId: string) => {
    setOrders(orders => orders.map(order => 
      order.id === orderId ? { ...order, status: 'cancelled' as const } : order
    ));
  };

  const getPriceColor = (change: number) => {
    return change >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getOrderStatusColor = (status: string) => {
    switch(status) {
      case 'filled': return 'text-green-400 bg-green-500/20';
      case 'pending': return 'text-yellow-400 bg-yellow-500/20';
      case 'cancelled': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const totalPortfolioValue = portfolio.reduce((sum, asset) => sum + asset.value_usd, 0);
  const portfolioChange24h = portfolio.reduce((sum, asset) => sum + (asset.value_usd * asset.change_24h / 100), 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link href="/wallet" className="inline-block mb-4">
          <motion.button
            className="px-4 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-sm"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            ← Back to Wallet
          </motion.button>
        </Link>
        
        <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 via-cyan-400 to-yellow-400 bg-clip-text text-transparent mb-2">
          📈 Cosmic Trading
        </h1>
        <p className="text-cyan-300">Advanced Multi-Dimensional Asset Exchange</p>
      </motion.header>

      {/* Portfolio Overview */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-purple-300">💼 Portfolio Overview</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="bg-gradient-to-br from-purple-500 to-violet-600 p-1 rounded-2xl">
            <div className="bg-black/70 rounded-xl p-4 text-center">
              <div className="text-2xl mb-1">💰</div>
              <div className="text-lg font-bold text-violet-300 break-all">${totalPortfolioValue.toLocaleString()}</div>
              <div className="text-sm text-violet-200">Total Value</div>
            </div>
          </div>
          <div className={`bg-gradient-to-br ${portfolioChange24h >= 0 ? 'from-green-500 to-emerald-600' : 'from-red-500 to-rose-600'} p-1 rounded-2xl`}>
            <div className="bg-black/70 rounded-xl p-4 text-center">
              <div className="text-2xl mb-1">{portfolioChange24h >= 0 ? '📈' : '📉'}</div>
              <div className={`text-lg font-bold ${portfolioChange24h >= 0 ? 'text-emerald-300' : 'text-rose-300'} break-all`}>
                ${Math.abs(portfolioChange24h).toLocaleString()}
              </div>
              <div className={`text-sm ${portfolioChange24h >= 0 ? 'text-emerald-200' : 'text-rose-200'}`}>24h P&L</div>
            </div>
          </div>
          <div className="bg-gradient-to-br from-blue-500 to-cyan-600 p-1 rounded-2xl">
            <div className="bg-black/70 rounded-xl p-4 text-center">
              <div className="text-2xl mb-1">📊</div>
              <div className="text-2xl font-bold text-cyan-300">{portfolio.length}</div>
              <div className="text-sm text-cyan-200">Assets</div>
            </div>
          </div>
          <div className="bg-gradient-to-br from-orange-500 to-red-600 p-1 rounded-2xl">
            <div className="bg-black/70 rounded-xl p-4 text-center">
              <div className="text-2xl mb-1">⚡</div>
              <div className="text-2xl font-bold text-orange-300">{orders.filter(o => o.status === 'pending').length}</div>
              <div className="text-sm text-orange-200">Open Orders</div>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="grid gap-8 lg:grid-cols-2">
        
        {/* Trading Interface */}
        <motion.div className="space-y-6" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
          
          {/* Market Selection */}
          <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-cyan-500/30">
            <h3 className="text-lg font-semibold mb-4 text-cyan-300">📊 Select Trading Pair</h3>
            <div className="space-y-2">
              {tradingPairs.map((pair, index) => (
                <motion.button
                  key={pair.symbol}
                  className={`w-full p-3 rounded-lg border transition-colors ${
                    selectedPair?.symbol === pair.symbol 
                      ? 'bg-cyan-600/30 border-cyan-400' 
                      : 'bg-black/30 border-gray-600 hover:border-cyan-500/50'
                  }`}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  onClick={() => setSelectedPair(pair)}
                >
                  <div className="flex justify-between items-center">
                    <div className="text-left">
                      <div className="font-semibold text-white">{pair.symbol}</div>
                      <div className="text-xs text-gray-400">Vol: {pair.volume_24h.toLocaleString()}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-yellow-300">${pair.price}</div>
                      <div className={`text-xs ${getPriceColor(pair.change_24h)}`}>
                        {pair.change_24h >= 0 ? '+' : ''}{pair.change_24h.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Order Form */}
          {selectedPair && (
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-green-500/30">
              <h3 className="text-lg font-semibold mb-4 text-green-300">💱 Place Order - {selectedPair.symbol}</h3>
              
              <div className="grid gap-4 md:grid-cols-2 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Order Type</label>
                  <div className="flex gap-2">
                    <button
                      className={`flex-1 px-3 py-2 rounded-lg text-sm ${
                        orderForm.type === 'buy' 
                          ? 'bg-green-600 text-white' 
                          : 'bg-gray-700 text-gray-300'
                      }`}
                      onClick={() => setOrderForm({...orderForm, type: 'buy'})}
                    >
                      📈 BUY
                    </button>
                    <button
                      className={`flex-1 px-3 py-2 rounded-lg text-sm ${
                        orderForm.type === 'sell' 
                          ? 'bg-red-600 text-white' 
                          : 'bg-gray-700 text-gray-300'
                      }`}
                      onClick={() => setOrderForm({...orderForm, type: 'sell'})}
                    >
                      📉 SELL
                    </button>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Order Mode</label>
                  <select
                    value={orderForm.order_type}
                    onChange={(e) => setOrderForm({...orderForm, order_type: e.target.value as 'market' | 'limit'})}
                    className="w-full px-3 py-2 bg-black/50 border border-green-500/30 rounded-lg text-white focus:border-green-400 focus:outline-none"
                  >
                    <option value="limit">Limit Order</option>
                    <option value="market">Market Order</option>
                  </select>
                </div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-2 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Amount ({selectedPair.base})</label>
                  <input
                    type="number"
                    step="0.001"
                    value={orderForm.amount}
                    onChange={(e) => setOrderForm({...orderForm, amount: e.target.value})}
                    placeholder="10.000"
                    className="w-full px-3 py-2 bg-black/50 border border-green-500/30 rounded-lg text-white placeholder-gray-500 focus:border-green-400 focus:outline-none"
                  />
                </div>
                
                {orderForm.order_type === 'limit' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Price ({selectedPair.quote})</label>
                    <input
                      type="number"
                      step="0.01"
                      value={orderForm.price}
                      onChange={(e) => setOrderForm({...orderForm, price: e.target.value})}
                      placeholder={selectedPair.price.toString()}
                      className="w-full px-3 py-2 bg-black/50 border border-green-500/30 rounded-lg text-white placeholder-gray-500 focus:border-green-400 focus:outline-none"
                    />
                  </div>
                )}
              </div>
              
              {orderForm.amount && (
                <div className="mb-4 p-3 bg-black/30 rounded-lg">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Total Cost:</span>
                    <span className="text-yellow-300">
                      {(parseFloat(orderForm.amount) * 
                        (orderForm.order_type === 'market' ? selectedPair.price : parseFloat(orderForm.price) || 0)
                      ).toFixed(6)} {selectedPair.quote}
                    </span>
                  </div>
                </div>
              )}
              
              <motion.button
                className={`w-full px-6 py-3 rounded-xl font-semibold ${
                  orderForm.type === 'buy'
                    ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                    : 'bg-gradient-to-r from-red-500 to-rose-600'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handlePlaceOrder}
              >
                {orderForm.type === 'buy' ? '📈 Place Buy Order' : '📉 Place Sell Order'}
              </motion.button>
            </div>
          )}
        </motion.div>

        {/* Right Column */}
        <motion.div className="space-y-6" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
          
          {/* Portfolio Assets */}
          <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-purple-500/30">
            <h3 className="text-lg font-semibold mb-4 text-purple-300">💼 Your Assets</h3>
            <div className="space-y-3">
              {portfolio.map((asset, index) => (
                <div key={asset.asset} className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                  <div>
                    <div className="font-semibold text-white">{asset.asset}</div>
                    <div className="text-sm text-gray-400">{asset.balance.toFixed(6)}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold text-yellow-300">${asset.value_usd.toFixed(2)}</div>
                    <div className={`text-xs ${getPriceColor(asset.change_24h)}`}>
                      {asset.change_24h >= 0 ? '+' : ''}{asset.change_24h.toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Open Orders */}
          <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-orange-500/30">
            <h3 className="text-lg font-semibold mb-4 text-orange-300">📋 Open Orders</h3>
            {orders.filter(order => order.status === 'pending').length === 0 ? (
              <div className="text-center py-4 text-gray-400">
                <div className="text-2xl mb-2">💤</div>
                <p>No open orders</p>
              </div>
            ) : (
              <div className="space-y-3">
                {orders.filter(order => order.status === 'pending').map((order, index) => (
                  <div key={order.id} className="p-3 bg-black/30 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <div className="font-semibold text-white">{order.pair}</div>
                        <div className={`text-sm ${order.type === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                          {order.type.toUpperCase()} {order.amount} @ ${order.price}
                        </div>
                      </div>
                      <motion.button
                        className="px-2 py-1 bg-red-600/30 hover:bg-red-600/50 rounded text-xs"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => cancelOrder(order.id)}
                      >
                        ❌ Cancel
                      </motion.button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Order History */}
      <motion.div className="mt-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-pink-300">📜 Trading History</h2>
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-pink-500/30">
          <div className="space-y-3">
            {orders.map((order, index) => (
              <div key={order.id} className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                <div>
                  <div className="font-semibold text-white">{order.pair}</div>
                  <div className="text-xs text-gray-400">{order.timestamp}</div>
                </div>
                <div className="text-center">
                  <div className={`text-sm ${order.type === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                    {order.type.toUpperCase()} {order.filled}/{order.amount}
                  </div>
                  <div className="text-xs text-yellow-300">${order.price}</div>
                </div>
                <div className="text-right">
                  <div className={`px-2 py-1 rounded text-xs ${getOrderStatusColor(order.status)}`}>
                    {order.status.toUpperCase()}
                  </div>
                  <div className="text-xs text-purple-300">
                    Dharma: {order.dharma_impact >= 0 ? '+' : ''}{order.dharma_impact}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}