#!/usr/bin/env python3
"""
🎯 ZION Multi-Algorithm Profitability Calculator
Vypočítá nejvýhodnější algoritmus based on current market prices
"""

import requests
import json
import time
from typing import Dict, List, Optional
import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

@dataclass
class AlgorithmData:
    name: str
    hashrate: float  # H/s
    power: float     # Watts
    coin: str
    pool_fee: float = 0.01  # 1%

@dataclass 
class CoinPrice:
    symbol: str
    price_usd: float
    change_24h: float

class ProfitabilityCalculator:
    def __init__(self, electricity_cost: float = 0.12):
        """
        Initialize calculator
        
        Args:
            electricity_cost: Cost per kWh in USD
        """
        self.electricity_cost = electricity_cost
        self.algorithms: List[AlgorithmData] = []
        self.coin_prices: Dict[str, CoinPrice] = {}
        
    def add_algorithm(self, name: str, hashrate: float, power: float, coin: str, pool_fee: float = 0.01):
        """Add algorithm benchmark data"""
        self.algorithms.append(AlgorithmData(name, hashrate, power, coin, pool_fee))
        
    def load_benchmark_data(self, benchmark_file: str = "benchmarks/results.csv"):
        """Load benchmark data from CSV file"""
        try:
            with open(benchmark_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        algo = parts[0]
                        hashrate_str = parts[1]
                        power_str = parts[2]
                        
                        # Parse hashrate
                        hashrate = self._parse_hashrate(hashrate_str)
                        power = self._parse_power(power_str)
                        
                        # Map algorithm to coin
                        coin_mapping = {
                            'kawpow': 'RVN',
                            'ergo': 'ERG', 
                            'ethash': 'ETC',
                            'octopus': 'CFX'
                        }
                        
                        if algo in coin_mapping and hashrate > 0:
                            self.add_algorithm(algo, hashrate, power, coin_mapping[algo])
                            
        except FileNotFoundError:
            print(f"⚠️  Benchmark file not found: {benchmark_file}")
            print("   Run benchmark first: ./benchmark-gpu.sh")
            
    def _parse_hashrate(self, hashrate_str: str) -> float:
        """Parse hashrate string to H/s"""
        if not hashrate_str or hashrate_str == "N/A":
            return 0.0
            
        # Remove units and get number
        import re
        numbers = re.findall(r'[\d.]+', hashrate_str)
        if not numbers:
            return 0.0
            
        value = float(numbers[0])
        
        # Convert to H/s based on unit
        if 'MH/s' in hashrate_str or 'Mh/s' in hashrate_str:
            value *= 1e6
        elif 'KH/s' in hashrate_str or 'Kh/s' in hashrate_str:
            value *= 1e3
        elif 'GH/s' in hashrate_str or 'Gh/s' in hashrate_str:
            value *= 1e9
            
        return value
        
    def _parse_power(self, power_str: str) -> float:
        """Parse power string to Watts"""
        if not power_str or power_str == "N/A":
            return 0.0
            
        import re
        numbers = re.findall(r'[\d.]+', power_str)
        if numbers:
            return float(numbers[0])
        return 0.0
        
    def fetch_coin_prices(self) -> bool:
        """Fetch current coin prices from CoinGecko API"""
        coins = list(set([algo.coin for algo in self.algorithms]))
        
        # CoinGecko coin IDs
        coin_ids = {
            'RVN': 'ravencoin',
            'ERG': 'ergo', 
            'ETC': 'ethereum-classic',
            'CFX': 'conflux-token'
        }
        
        try:
            ids = [coin_ids.get(coin, coin.lower()) for coin in coins if coin in coin_ids]
            
            if not ids:
                return False
                
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': ','.join(ids),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Map back to coin symbols
            reverse_mapping = {v: k for k, v in coin_ids.items()}
            
            for coin_id, price_data in data.items():
                symbol = reverse_mapping.get(coin_id, coin_id.upper())
                self.coin_prices[symbol] = CoinPrice(
                    symbol=symbol,
                    price_usd=price_data['usd'],
                    change_24h=price_data.get('usd_24h_change', 0.0)
                )
                
            return True
            
        except requests.RequestException as e:
            print(f"❌ Failed to fetch prices: {e}")
            return False
            
    def calculate_daily_profit(self, algo: AlgorithmData) -> Dict:
        """Calculate daily profit for algorithm"""
        if algo.coin not in self.coin_prices:
            return {}
            
        coin_price = self.coin_prices[algo.coin]
        
        # Network difficulty and block reward (estimated)
        network_stats = self._get_network_stats(algo.coin, algo.name)
        
        if not network_stats:
            return {}
            
        # Calculate daily coins mined
        network_hashrate = network_stats['network_hashrate']
        block_reward = network_stats['block_reward']
        blocks_per_day = network_stats['blocks_per_day']
        
        # Your share of network hashrate
        hashrate_share = algo.hashrate / network_hashrate if network_hashrate > 0 else 0
        
        # Daily coins before pool fee
        daily_coins_gross = hashrate_share * block_reward * blocks_per_day
        daily_coins_net = daily_coins_gross * (1 - algo.pool_fee)
        
        # Daily revenue and costs
        daily_revenue = daily_coins_net * coin_price.price_usd
        daily_electricity = (algo.power / 1000) * 24 * self.electricity_cost
        daily_profit = daily_revenue - daily_electricity
        
        # Efficiency metrics
        efficiency = daily_profit / algo.power if algo.power > 0 else 0
        roi_days = abs(daily_electricity / daily_profit) if daily_profit != 0 else float('inf')
        
        return {
            'algorithm': algo.name,
            'coin': algo.coin,
            'hashrate': algo.hashrate,
            'power': algo.power,
            'daily_coins': daily_coins_net,
            'daily_revenue': daily_revenue,
            'daily_electricity': daily_electricity,
            'daily_profit': daily_profit,
            'efficiency': efficiency,
            'roi_days': roi_days,
            'coin_price': coin_price.price_usd,
            'price_change_24h': coin_price.change_24h
        }
        
    def _get_network_stats(self, coin: str, algorithm: str) -> Optional[Dict]:
        """Get network statistics for coin/algorithm"""
        
        # Estimated network stats (in production, fetch from APIs)
        stats = {
            'RVN': {  # KawPow
                'network_hashrate': 50e12,  # 50 TH/s
                'block_reward': 5000,
                'blocks_per_day': 1440  # 1 minute blocks
            },
            'ERG': {  # Autolykos2
                'network_hashrate': 20e12,  # 20 TH/s  
                'block_reward': 51,
                'blocks_per_day': 720  # 2 minute blocks
            },
            'ETC': {  # Ethash
                'network_hashrate': 180e12,  # 180 TH/s
                'block_reward': 3.2,
                'blocks_per_day': 6400  # ~13.5 second blocks
            },
            'CFX': {  # Octopus
                'network_hashrate': 5e12,   # 5 TH/s
                'block_reward': 2,
                'blocks_per_day': 86400  # 1 second blocks
            }
        }
        
        return stats.get(coin)
        
    def calculate_all_profits(self) -> List[Dict]:
        """Calculate profits for all algorithms"""
        results = []
        
        for algo in self.algorithms:
            profit_data = self.calculate_daily_profit(algo)
            if profit_data:
                results.append(profit_data)
                
        # Sort by daily profit (descending)
        results.sort(key=lambda x: x['daily_profit'], reverse=True)
        return results
        
    def print_results(self, results: List[Dict]):
        """Print formatted profitability results"""
        if not results:
            print("❌ No profitability data available")
            print("   1. Run benchmark: ./benchmark-gpu.sh")  
            print("   2. Check internet connection for price data")
            return
            
        print("💰 ZION Multi-Algorithm Profitability Report")
        print("=" * 60)
        print(f"⚡ Electricity Cost: ${self.electricity_cost:.3f}/kWh")
        print(f"🕒 Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("📊 Profitability Ranking:")
        print("-" * 100)
        print(f"{'Rank':<4} {'Algorithm':<10} {'Coin':<5} {'Daily Profit':<12} {'Revenue':<10} {'Power':<8} {'Efficiency':<12}")
        print("-" * 100)
        
        for i, result in enumerate(results, 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{rank_emoji}{i:<3} {result['algorithm']:<10} "
                  f"{result['coin']:<5} ${result['daily_profit']:<11.2f} "
                  f"${result['daily_revenue']:<9.2f} {result['power']:<7.0f}W "
                  f"${result['efficiency']:<11.4f}")
                  
        print("-" * 100)
        print()
        
        # Detailed breakdown for top algorithm
        if results:
            top = results[0] 
            print(f"🎯 Top Performer: {top['algorithm'].upper()} ({top['coin']})")
            print(f"   💎 Daily Coins: {top['daily_coins']:.6f} {top['coin']}")
            print(f"   💵 Coin Price: ${top['coin_price']:.4f} ({top['price_change_24h']:+.2f}%)")
            print(f"   ⚡ Power Usage: {top['power']:.0f}W")
            print(f"   💰 Net Profit: ${top['daily_profit']:.2f}/day")
            print(f"   📈 Monthly Estimate: ${top['daily_profit'] * 30:.2f}")
            print()
            
        # Recommendations
        print("💡 Recommendations:")
        if len(results) >= 2:
            profit_diff = results[0]['daily_profit'] - results[1]['daily_profit']
            print(f"   • Switch to {results[0]['algorithm']} for ${profit_diff:.2f}/day more profit")
            
        for result in results:
            if result['daily_profit'] < 0:
                print(f"   ⚠️  {result['algorithm']} is losing money (${result['daily_profit']:.2f}/day)")
                
        print("   • Monitor prices regularly for optimal switching")
        print("   • Consider dual mining during low profitability periods")

def main():
    parser = argparse.ArgumentParser(description='🎯 ZION Multi-Algorithm Profitability Calculator')
    parser.add_argument('--electricity', type=float, default=0.12, 
                       help='Electricity cost per kWh in USD (default: 0.12)')
    parser.add_argument('--benchmark-file', default='benchmarks/results.csv',
                       help='Benchmark results file (default: benchmarks/results.csv)')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calc = ProfitabilityCalculator(electricity_cost=args.electricity)
    
    # Load benchmark data
    calc.load_benchmark_data(args.benchmark_file)
    
    if not calc.algorithms:
        print("❌ No benchmark data found!")
        print("   Run: ./benchmark-gpu.sh")
        return
        
    # Fetch current prices  
    print("🔄 Fetching current coin prices...")
    if not calc.fetch_coin_prices():
        print("❌ Failed to fetch coin prices")
        print("   Check internet connection and try again")
        return
        
    # Calculate and display results
    results = calc.calculate_all_profits()
    calc.print_results(results)
    
    # Save results
    output_file = "benchmarks/profitability-report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"💾 Detailed results saved: {output_file}")

if __name__ == "__main__":
    main()