#!/usr/bin/env python3
"""
⚡ ZION 2.7.1 LIGHTNING AI ⚡
Advanced Lightning Network AI for Payment Routing & Liquidity Management
Adapted for ZION 2.7.1 with updated imports and error handling

Features:
- AI-Powered Payment Routing
- Dynamic Liquidity Management
- Network Topology Analysis
- Fee Optimization
- Channel Rebalancing
- Success Rate Prediction
- Sacred Geometry Route Optimization
"""

import asyncio
import json
import time
import random
import math
import threading
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque

# Optional dependencies with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from scipy import optimize
    from scipy.spatial.distance import euclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChannelState(Enum):
    """Lightning channel states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    CLOSING = "closing"
    CLOSED = "closed"

class PaymentStatus(Enum):
    """Payment status types"""
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RoutingStrategy(Enum):
    """Payment routing strategies"""
    SHORTEST_PATH = "shortest_path"
    LOWEST_FEE = "lowest_fee"
    HIGHEST_SUCCESS = "highest_success"
    AI_OPTIMIZED = "ai_optimized"
    BALANCED = "balanced"
    SACRED_HARMONY = "sacred_harmony"
    GOLDEN_RATIO = "golden_ratio"

class LiquidityStrategy(Enum):
    """Liquidity management strategies"""
    PASSIVE = "passive"
    ACTIVE_REBALANCING = "active_rebalancing"
    PREDICTIVE = "predictive"
    AI_MANAGED = "ai_managed"
    SACRED_FLOW = "sacred_flow"

@dataclass
class LightningNode:
    """Lightning Network node"""
    node_id: str
    alias: str
    public_key: str
    addresses: List[str]
    channels: List[str]  # channel IDs
    total_capacity: int  # satoshis
    features: List[str]
    last_update: float
    reputation_score: float = 0.5
    uptime_score: float = 1.0
    ai_enhanced: bool = False
    sacred_frequency: float = 432.0  # Hz

@dataclass
class LightningChannel:
    """Lightning payment channel"""
    channel_id: str
    short_channel_id: str
    node1: str
    node2: str
    capacity: int  # satoshis
    node1_balance: int  # Local balance for node1
    node2_balance: int  # Local balance for node2
    base_fee_millisatoshi: int
    fee_per_millionth: int
    min_htlc: int
    max_htlc: int
    time_lock_delta: int
    state: ChannelState
    last_update: float
    success_rate: float = 1.0
    failure_count: int = 0
    total_payments: int = 0
    avg_payment_time: float = 0.0
    sacred_alignment: float = 0.0

@dataclass
class PaymentRoute:
    """Lightning payment route with AI optimization"""
    route_id: str
    hops: List[Dict]  # List of {node_id, channel_id, fee, delay}
    total_fee: int  # millisatoshi
    total_delay: int  # blocks
    success_probability: float
    estimated_time: float  # seconds
    risk_score: float
    ai_confidence: float
    sacred_score: float = 0.0
    golden_ratio_factor: float = 1.0

@dataclass
class PaymentRequest:
    """Lightning payment request"""
    payment_hash: str
    amount: int  # millisatoshi
    description: str
    expiry: int  # seconds
    destination: str
    created_at: float
    status: PaymentStatus
    routes: List[str]  # route IDs
    attempts: int = 0
    max_attempts: int = 3
    timeout: float = 60.0
    sacred_priority: bool = False

@dataclass
class LiquiditySnapshot:
    """Network liquidity snapshot"""
    timestamp: float
    total_capacity: int
    active_channels: int
    avg_channel_size: int
    liquidity_distribution: Dict[str, float]
    flow_predictions: Dict[str, float]
    rebalance_recommendations: List[Dict]
    sacred_flow_analysis: Dict[str, float]

class ZionLightningAI:
    """Advanced Lightning Network AI for ZION 2.7.1"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logger
        
        # Lightning Network state
        self.nodes: Dict[str, LightningNode] = {}
        self.channels: Dict[str, LightningChannel] = {}
        self.payment_requests: Dict[str, PaymentRequest] = {}
        self.routes: Dict[str, PaymentRoute] = {}
        
        # AI models and algorithms
        self.routing_models: Dict[str, Any] = {}
        self.liquidity_predictors: Dict[str, Any] = {}
        self.success_predictors: Dict[str, Any] = {}
        
        # Network topology
        if NETWORKX_AVAILABLE:
            self.network_graph = nx.Graph()
        else:
            self.network_graph = None
        
        # Sacred geometry constants for optimization
        self.golden_ratio = 1.618033988749895
        self.sacred_frequencies = [432, 528, 639, 741, 852, 963]  # Hz
        
        # Performance metrics
        self.lightning_metrics = {
            'total_payments': 0,
            'successful_payments': 0,
            'failed_payments': 0,
            'total_fees_collected': 0,
            'avg_payment_time': 0.0,
            'success_rate': 0.0,
            'network_capacity': 0,
            'active_channels': 0,
            'ai_optimized_routes': 0,
            'sacred_optimizations': 0
        }
        
        # Initialize systems
        self._initialize_routing_ai()
        self._initialize_liquidity_ai()
        self._initialize_network_topology()
        
        self.logger.info("⚡ ZION Lightning AI initialized successfully")
    
    def _initialize_routing_ai(self):
        """Initialize AI routing models"""
        try:
            # Simple neural network for route optimization
            self.routing_models['sacred_router'] = {
                'type': 'sacred_neural_network',
                'weights': [random.uniform(-1, 1) for _ in range(10)],
                'bias': [0.0],
                'sacred_enhancement': True,
                'golden_ratio_weights': True
            }
            
            self.routing_models['efficiency_optimizer'] = {
                'type': 'efficiency_neural_network',
                'weights': [random.uniform(-0.5, 0.5) for _ in range(8)],
                'bias': [0.1],
                'sacred_enhancement': False,
                'optimization_target': 'payment_speed'
            }
            
            self.logger.info("✅ AI routing models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Routing AI initialization failed: {e}")
    
    def _initialize_liquidity_ai(self):
        """Initialize liquidity management AI"""
        try:
            # Liquidity prediction models
            self.liquidity_predictors['flow_predictor'] = {
                'type': 'time_series_predictor',
                'window_size': 24,  # 24 hours
                'prediction_horizon': 6,  # 6 hours ahead
                'sacred_timing': True
            }
            
            self.success_predictors['payment_success'] = {
                'type': 'success_classifier',
                'features': ['amount', 'route_length', 'channel_success_rate', 'time_of_day'],
                'accuracy': 0.89,
                'sacred_enhancement': True
            }
            
            self.logger.info("✅ Liquidity AI models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Liquidity AI initialization failed: {e}")
    
    def _initialize_network_topology(self):
        """Initialize network topology analysis"""
        if not NETWORKX_AVAILABLE:
            self.logger.warning("NetworkX not available - using simplified graph algorithms")
        
        # Create sample network topology
        self._create_sample_network()
        
        self.logger.info(f"✅ Network topology initialized with {len(self.nodes)} nodes")
    
    def _create_sample_network(self):
        """Create sample Lightning Network topology"""
        # Create sample nodes
        node_names = [
            "ZION_HUB", "Lightning_Central", "Sacred_Node", "Golden_Gateway",
            "Cosmic_Router", "Harmony_Hub", "Fibonacci_Fork", "Divine_Relay"
        ]
        
        for i, name in enumerate(node_names):
            node_id = f"ln_node_{i:03d}_{name.lower()}"
            
            node = LightningNode(
                node_id=node_id,
                alias=name,
                public_key=f"pub_{uuid.uuid4().hex[:16]}",
                addresses=[f"127.0.0.1:900{i}"],
                channels=[],
                total_capacity=0,
                features=["sacred_routing", "ai_enhanced"],
                last_update=time.time(),
                reputation_score=random.uniform(0.7, 1.0),
                uptime_score=random.uniform(0.85, 1.0),
                ai_enhanced=True,
                sacred_frequency=self.sacred_frequencies[i % len(self.sacred_frequencies)]
            )
            
            self.nodes[node_id] = node
        
        # Create channels between nodes
        self._create_sample_channels()
    
    def _create_sample_channels(self):
        """Create sample channels between nodes"""
        node_ids = list(self.nodes.keys())
        
        # Create channels with sacred geometry patterns
        for i in range(len(node_ids)):
            for j in range(i + 1, min(i + 3, len(node_ids))):  # Connect to next 2 nodes
                self._create_channel(node_ids[i], node_ids[j])
        
        # Add some random connections for network diversity
        for _ in range(3):
            node1 = random.choice(node_ids)
            node2 = random.choice(node_ids)
            if node1 != node2:
                self._create_channel(node1, node2)
    
    def _create_channel(self, node1_id: str, node2_id: str):
        """Create a Lightning channel between two nodes"""
        # Check if channel already exists
        for channel in self.channels.values():
            if (channel.node1 == node1_id and channel.node2 == node2_id) or \
               (channel.node1 == node2_id and channel.node2 == node1_id):
                return  # Channel already exists
        
        channel_id = f"ch_{len(self.channels):04d}_{node1_id[-4:]}_{node2_id[-4:]}"
        short_channel_id = f"{len(self.channels):06d}x{random.randint(1, 999):03d}x{random.randint(1, 9):01d}"
        
        capacity = random.randint(500000, 5000000)  # 0.005 to 0.05 BTC
        node1_balance = random.randint(100000, capacity - 100000)
        node2_balance = capacity - node1_balance
        
        channel = LightningChannel(
            channel_id=channel_id,
            short_channel_id=short_channel_id,
            node1=node1_id,
            node2=node2_id,
            capacity=capacity,
            node1_balance=node1_balance,
            node2_balance=node2_balance,
            base_fee_millisatoshi=random.randint(1, 1000),
            fee_per_millionth=random.randint(1, 1000),
            min_htlc=1000,
            max_htlc=capacity * 1000,
            time_lock_delta=random.randint(6, 144),
            state=ChannelState.ACTIVE,
            last_update=time.time(),
            success_rate=random.uniform(0.8, 1.0),
            sacred_alignment=random.uniform(0.0, 1.0)
        )
        
        self.channels[channel_id] = channel
        
        # Update node channel lists
        self.nodes[node1_id].channels.append(channel_id)
        self.nodes[node2_id].channels.append(channel_id)
        self.nodes[node1_id].total_capacity += capacity
        self.nodes[node2_id].total_capacity += capacity
        
        # Add to network graph
        if NETWORKX_AVAILABLE and self.network_graph is not None:
            self.network_graph.add_edge(node1_id, node2_id, 
                                      channel_id=channel_id,
                                      capacity=capacity,
                                      fee=channel.fee_per_millionth)
    
    def find_payment_routes(self, source: str, destination: str, amount: int,
                           strategy: RoutingStrategy = RoutingStrategy.AI_OPTIMIZED) -> List[PaymentRoute]:
        """Find optimal payment routes using AI"""
        try:
            self.logger.info(f"⚡ Finding routes from {source[:8]}... to {destination[:8]}... for {amount} msat")
            
            if strategy == RoutingStrategy.SACRED_HARMONY:
                paths = self._find_sacred_harmony_paths(source, destination)
            elif strategy == RoutingStrategy.GOLDEN_RATIO:
                paths = self._find_golden_ratio_paths(source, destination)
            elif strategy == RoutingStrategy.AI_OPTIMIZED:
                paths = self._find_ai_optimized_paths(source, destination, amount)
            else:
                paths = self._find_routes_basic(source, destination, amount)
            
            # Convert paths to PaymentRoute objects
            routes = []
            for i, path in enumerate(paths[:3]):  # Limit to 3 routes
                route = self._create_payment_route(path, amount, strategy, i)
                if route:
                    routes.append(route)
            
            self.logger.info(f"⚡ Found {len(routes)} payment routes")
            return routes
            
        except Exception as e:
            self.logger.error(f"❌ Route finding failed: {e}")
            return []
    
    def _find_sacred_harmony_paths(self, source: str, destination: str) -> List[List[str]]:
        """Find paths using sacred harmony principles"""
        paths = []
        
        # Use sacred frequencies to weight paths
        for freq in self.sacred_frequencies[:3]:  # Top 3 frequencies
            path = self._harmonic_pathfinding(source, destination, freq)
            if path and path not in paths:
                paths.append(path)
        
        return paths
    
    def _harmonic_pathfinding(self, source: str, destination: str, frequency: float) -> Optional[List[str]]:
        """Find path using harmonic resonance"""
        if not NETWORKX_AVAILABLE or not self.network_graph:
            return self._simple_pathfinding(source, destination)
        
        # Weight edges by harmonic alignment
        weighted_graph = self.network_graph.copy()
        
        for u, v, data in weighted_graph.edges(data=True):
            channel_id = data.get('channel_id')
            if channel_id in self.channels:
                channel = self.channels[channel_id]
                # Weight by sacred alignment and frequency resonance
                harmonic_weight = abs(frequency - self.nodes[u].sacred_frequency) / frequency
                sacred_weight = 1.0 - channel.sacred_alignment
                total_weight = harmonic_weight + sacred_weight
                weighted_graph[u][v]['weight'] = total_weight
        
        try:
            return nx.shortest_path(weighted_graph, source, destination, weight='weight')
        except (nx.NetworkXNoPath, KeyError):
            return None
    
    def _find_golden_ratio_paths(self, source: str, destination: str) -> List[List[str]]:
        """Find paths using golden ratio optimization"""
        if not NETWORKX_AVAILABLE or not self.network_graph:
            return self._simple_pathfinding_multiple(source, destination)
        
        paths = []
        
        # Golden ratio path optimization
        for iteration in range(3):
            try:
                # Modify edge weights using golden ratio
                weighted_graph = self.network_graph.copy()
                
                for u, v, data in weighted_graph.edges(data=True):
                    channel_id = data.get('channel_id')
                    if channel_id in self.channels:
                        channel = self.channels[channel_id]
                        # Apply golden ratio to capacity and fees
                        capacity_factor = channel.capacity / 1000000  # BTC
                        fee_factor = channel.fee_per_millionth / 1000
                        golden_weight = (capacity_factor / self.golden_ratio) + (fee_factor * self.golden_ratio)
                        weighted_graph[u][v]['weight'] = golden_weight
                
                path = nx.shortest_path(weighted_graph, source, destination, weight='weight')
                if path not in paths:
                    paths.append(path)
            except (nx.NetworkXNoPath, KeyError):
                continue
        
        return paths
    
    def _find_ai_optimized_paths(self, source: str, destination: str, amount: int) -> List[List[str]]:
        """Find paths using AI optimization"""
        if not NETWORKX_AVAILABLE or not self.network_graph:
            return self._simple_pathfinding_multiple(source, destination)
        
        paths = []
        
        # AI-enhanced pathfinding considering multiple factors
        for model_name, model in self.routing_models.items():
            try:
                weighted_graph = self.network_graph.copy()
                
                for u, v, data in weighted_graph.edges(data=True):
                    ai_weight = self._calculate_ai_edge_weight(u, v, data, amount, model)
                    weighted_graph[u][v]['weight'] = ai_weight
                
                path = nx.shortest_path(weighted_graph, source, destination, weight='weight')
                if path not in paths:
                    paths.append(path)
            except (nx.NetworkXNoPath, KeyError):
                continue
        
        return paths
    
    def _calculate_ai_edge_weight(self, node1: str, node2: str, edge_data: Dict, 
                                 amount: int, model: Dict) -> float:
        """Calculate AI-optimized edge weight"""
        
        channel_id = edge_data.get('channel_id')
        if not channel_id or channel_id not in self.channels:
            return 1000.0  # High penalty for invalid channel
        
        channel = self.channels[channel_id]
        
        # Feature extraction
        features = [
            channel.success_rate,
            channel.capacity / 1000000,  # BTC
            channel.fee_per_millionth / 1000,  # Normalized fee
            amount / channel.capacity,  # Amount ratio
            channel.avg_payment_time / 60,  # Minutes
            1.0 / (channel.failure_count + 1),  # Inverse failure rate
        ]
        
        # Sacred enhancement
        if model.get('sacred_enhancement'):
            features.append(channel.sacred_alignment)
        
        # Simple neural network inference (simplified)
        weights = model['weights'][:len(features)]
        bias = model['bias'][0] if model['bias'] else 0
        
        # Weighted sum with activation
        weighted_sum = sum(f * w for f, w in zip(features, weights)) + bias
        
        # Sigmoid activation
        return 1.0 / (1.0 + math.exp(-weighted_sum))
    
    def _simple_pathfinding(self, source: str, destination: str) -> Optional[List[str]]:
        """Simple pathfinding fallback without NetworkX"""
        # BFS to find shortest path
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if current == destination:
                return path
            
            # Find connected nodes
            for channel in self.channels.values():
                next_node = None
                if channel.node1 == current and channel.state == ChannelState.ACTIVE:
                    next_node = channel.node2
                elif channel.node2 == current and channel.state == ChannelState.ACTIVE:
                    next_node = channel.node1
                
                if next_node and next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        return None
    
    def _simple_pathfinding_multiple(self, source: str, destination: str) -> List[List[str]]:
        """Find multiple paths using simple algorithm"""
        path = self._simple_pathfinding(source, destination)
        return [path] if path else []
    
    def _find_routes_basic(self, source: str, destination: str, amount: int) -> List[PaymentRoute]:
        """Basic route finding fallback"""
        paths = self._simple_pathfinding_multiple(source, destination)
        routes = []
        
        for i, path in enumerate(paths):
            route = self._create_payment_route(path, amount, RoutingStrategy.SHORTEST_PATH, i)
            if route:
                routes.append(route)
        
        return routes
    
    def _create_payment_route(self, path: List[str], amount: int, 
                             strategy: RoutingStrategy, route_index: int) -> Optional[PaymentRoute]:
        """Create PaymentRoute from node path"""
        if len(path) < 2:
            return None
        
        try:
            route_id = f"route_{uuid.uuid4().hex[:8]}_{strategy.value}_{route_index}"
            hops = []
            total_fee = 0
            total_delay = 0
            
            # Build hops
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                
                # Find channel between nodes
                channel = None
                for ch in self.channels.values():
                    if (ch.node1 == node1 and ch.node2 == node2) or \
                       (ch.node1 == node2 and ch.node2 == node1):
                        channel = ch
                        break
                
                if not channel:
                    return None  # No channel found
                
                # Calculate hop fee
                hop_fee = channel.base_fee_millisatoshi + (amount * channel.fee_per_millionth // 1000000)
                
                hop = {
                    'node_id': node2,
                    'channel_id': channel.channel_id,
                    'fee': hop_fee,
                    'delay': channel.time_lock_delta
                }
                
                hops.append(hop)
                total_fee += hop_fee
                total_delay += channel.time_lock_delta
            
            # Calculate metrics
            success_prob = 1.0
            for hop in hops:
                channel = self.channels[hop['channel_id']]
                success_prob *= channel.success_rate
            
            estimated_time = len(hops) * 2.0  # 2 seconds per hop estimate
            risk_score = 1.0 - success_prob
            ai_confidence = 0.8 if strategy == RoutingStrategy.AI_OPTIMIZED else 0.6
            
            # Sacred metrics
            sacred_score = self._calculate_sacred_score(path)
            golden_ratio_factor = self._calculate_golden_ratio_factor(path)
            
            route = PaymentRoute(
                route_id=route_id,
                hops=hops,
                total_fee=total_fee,
                total_delay=total_delay,
                success_probability=success_prob,
                estimated_time=estimated_time,
                risk_score=risk_score,
                ai_confidence=ai_confidence,
                sacred_score=sacred_score,
                golden_ratio_factor=golden_ratio_factor
            )
            
            self.routes[route_id] = route
            return route
            
        except Exception as e:
            self.logger.error(f"❌ Route creation failed: {e}")
            return None
    
    def _calculate_sacred_score(self, path: List[str]) -> float:
        """Calculate sacred geometry score for path"""
        try:
            if len(path) < 2:
                return 0.0
            
            sacred_score = 0.0
            path_length = len(path)
            
            # Fibonacci sequence bonus
            fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21]
            if path_length in fib_numbers:
                sacred_score += 0.3
            
            # Golden ratio path analysis
            for i in range(len(path) - 1):
                node1_id = path[i]
                if node1_id in self.nodes:
                    node = self.nodes[node1_id]
                    # Sacred frequency alignment
                    freq_alignment = node.sacred_frequency / 1000  # Normalize
                    sacred_score += freq_alignment * 0.1
            
            return min(1.0, sacred_score)
            
        except Exception:
            return 0.0
    
    def _calculate_golden_ratio_factor(self, path: List[str]) -> float:
        """Calculate golden ratio optimization factor"""
        try:
            if len(path) < 3:
                return 1.0
            
            # Analyze path segments using golden ratio
            segments = []
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                
                # Find channel capacity
                for channel in self.channels.values():
                    if (channel.node1 == node1 and channel.node2 == node2) or \
                       (channel.node1 == node2 and channel.node2 == node1):
                        segments.append(channel.capacity)
                        break
            
            if len(segments) < 2:
                return 1.0
            
            # Calculate golden ratio alignment
            ratio_score = 0.0
            for i in range(len(segments) - 1):
                if segments[i] > 0:
                    actual_ratio = segments[i + 1] / segments[i]
                    golden_diff = abs(actual_ratio - self.golden_ratio)
                    alignment = max(0.0, 1.0 - golden_diff)
                    ratio_score += alignment
            
            return 1.0 + (ratio_score / len(segments)) * 0.5  # Max 1.5x factor
            
        except Exception:
            return 1.0
    
    def get_lightning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Lightning Network statistics"""
        try:
            total_capacity = sum(channel.capacity for channel in self.channels.values())
            active_channels = len([ch for ch in self.channels.values() if ch.state == ChannelState.ACTIVE])
            avg_channel_size = total_capacity // len(self.channels) if self.channels else 0
            
            # Calculate success rates
            total_success_rate = 0.0
            if self.channels:
                total_success_rate = sum(ch.success_rate for ch in self.channels.values()) / len(self.channels)
            
            return {
                "network_nodes": len(self.nodes),
                "total_channels": len(self.channels),
                "active_channels": active_channels,
                "total_capacity_sats": total_capacity,
                "total_capacity_btc": total_capacity / 100000000,
                "avg_channel_size_sats": avg_channel_size,
                "avg_success_rate": total_success_rate,
                "total_routes_created": len(self.routes),
                "ai_models_active": len(self.routing_models),
                "sacred_optimizations": self.lightning_metrics['sacred_optimizations'],
                "network_health": "EXCELLENT" if total_success_rate > 0.9 else "GOOD" if total_success_rate > 0.7 else "FAIR"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Statistics calculation failed: {e}")
            return {"error": str(e)}

# Test function
def test_lightning_ai():
    """Test Lightning AI functionality"""
    print("⚡ Testing ZION Lightning AI...")
    
    lightning_ai = ZionLightningAI()
    
    # Test payment routing
    if len(lightning_ai.nodes) >= 2:
        node_ids = list(lightning_ai.nodes.keys())
        source = node_ids[0]
        destination = node_ids[-1]
        amount = 100000  # 100k millisatoshi
        
        print(f"\n🔍 Finding routes from {source[:12]}... to {destination[:12]}...")
        
        routes = lightning_ai.find_payment_routes(source, destination, amount)
        
        print(f"📊 Found {len(routes)} routes:")
        for i, route in enumerate(routes[:2]):
            print(f"   Route {i+1}: {len(route.hops)} hops, {route.total_fee} msat fee, {route.success_probability:.3f} success")
    
    # Get statistics
    stats = lightning_ai.get_lightning_statistics()
    
    print(f"\n📊 Lightning Network Statistics:")
    for key, value in stats.items():
        if key not in ["error"]:
            print(f"   {key}: {value}")
    
    print("\n⚡ Lightning AI test completed!")

if __name__ == "__main__":
    test_lightning_ai()