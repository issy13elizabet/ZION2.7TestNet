"""
ZION Rainbow Bridge - Advanced Multi-Chain Integration
Cross-chain bridges for Solana, Stellar, Cardano, Tron and other major networks
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal
import aiohttp
from aiohttp import web
import websockets
from solana.rpc.async_api import AsyncClient as SolanaClient
from stellar_sdk import Asset, Keypair, Network, Server, TransactionBuilder
from stellar_sdk.account import Account
import tronpy
from tronpy.providers import HTTPProvider
import subprocess
import os


class ChainType(Enum):
    """Supported blockchain networks"""
    ZION = "zion"
    SOLANA = "solana"  
    STELLAR = "stellar"
    CARDANO = "cardano"
    TRON = "tron"
    ETHEREUM = "ethereum"
    BINANCE = "binance"
    POLYGON = "polygon"


class BridgeStatus(Enum):
    """Bridge transaction status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BridgeTransaction:
    """Cross-chain bridge transaction"""
    bridge_id: str
    from_chain: ChainType
    to_chain: ChainType
    from_address: str
    to_address: str
    from_txid: str
    to_txid: Optional[str]
    amount: Decimal
    fee: Decimal
    status: BridgeStatus
    created_at: int
    completed_at: Optional[int]
    confirmations: int
    required_confirmations: int


@dataclass
class ChainConfig:
    """Chain-specific configuration"""
    name: str
    rpc_url: str
    network_id: str
    confirmations_required: int
    min_amount: Decimal
    max_amount: Decimal
    bridge_fee: Decimal
    contract_address: Optional[str] = None
    private_key: Optional[str] = None


class ZionRainbowBridge:
    """Advanced multi-chain bridge system"""
    
    def __init__(self, 
                 bridge_private_key: str,
                 zion_rpc_url: str = "http://localhost:18089",
                 web_port: int = 9000,
                 websocket_port: int = 9001):
        
        # Bridge configuration
        self.bridge_private_key = bridge_private_key
        self.zion_rpc_url = zion_rpc_url
        self.web_port = web_port
        self.websocket_port = websocket_port
        
        # Chain configurations
        self.chains: Dict[ChainType, ChainConfig] = {}
        self._initialize_chain_configs()
        
        # Bridge state
        self.active_bridges: Dict[str, BridgeTransaction] = {}
        self.completed_bridges: List[BridgeTransaction] = []
        self.chain_clients: Dict[ChainType, Any] = {}
        
        # Services
        self.web_server = None
        self.websocket_server = None
        self.running = False
        
        # Statistics
        self.bridge_stats = {
            'total_bridges': 0,
            'successful_bridges': 0,
            'failed_bridges': 0,
            'total_volume': Decimal('0'),
            'chains_connected': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger('zion_rainbow_bridge')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [BRIDGE] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _initialize_chain_configs(self):
        """Initialize supported chain configurations"""
        # ZION (native chain)
        self.chains[ChainType.ZION] = ChainConfig(
            name="ZION Network",
            rpc_url=self.zion_rpc_url,
            network_id="mainnet",
            confirmations_required=6,
            min_amount=Decimal('1'),
            max_amount=Decimal('1000000'),
            bridge_fee=Decimal('0.1')
        )
        
        # Solana
        self.chains[ChainType.SOLANA] = ChainConfig(
            name="Solana",
            rpc_url=os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),
            network_id="mainnet-beta",
            confirmations_required=32,
            min_amount=Decimal('0.1'),
            max_amount=Decimal('100000'),
            bridge_fee=Decimal('0.01'),
            contract_address="ZionBridgeProgram1111111111111111111111111"
        )
        
        # Stellar
        self.chains[ChainType.STELLAR] = ChainConfig(
            name="Stellar",
            rpc_url="https://horizon.stellar.org",
            network_id="mainnet",
            confirmations_required=1,
            min_amount=Decimal('1'),
            max_amount=Decimal('50000'),
            bridge_fee=Decimal('0.5'),
            contract_address="ZION" # Stellar asset code
        )
        
        # Cardano
        self.chains[ChainType.CARDANO] = ChainConfig(
            name="Cardano",
            rpc_url=os.getenv('CARDANO_RPC_URL', 'https://cardano-mainnet.blockfrost.io/api/v0'),
            network_id="mainnet",
            confirmations_required=5,
            min_amount=Decimal('2'),
            max_amount=Decimal('25000'),
            bridge_fee=Decimal('1'),
            contract_address="addr1_cardano_zion_bridge_contract"
        )
        
        # Tron
        self.chains[ChainType.TRON] = ChainConfig(
            name="Tron",
            rpc_url="https://api.trongrid.io",
            network_id="mainnet",
            confirmations_required=19,
            min_amount=Decimal('10'),
            max_amount=Decimal('100000'),
            bridge_fee=Decimal('5'),
            contract_address="TZionBridge123456789ABCDEFabcdef"
        )
    
    async def start(self):
        """Start rainbow bridge services"""
        self.running = True
        self.logger.info("Starting ZION Rainbow Bridge...")
        
        # Initialize chain clients
        await self._initialize_chain_clients()
        
        # Start web API server
        await self._start_web_server()
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        # Start background monitoring
        asyncio.create_task(self._transaction_monitor_loop())
        asyncio.create_task(self._chain_sync_loop())
        asyncio.create_task(self._stats_update_loop())
        
        self.logger.info(f"Rainbow Bridge started on ports {self.web_port} (HTTP) and {self.websocket_port} (WS)")
    
    async def stop(self):
        """Stop bridge services"""
        self.running = False
        
        # Close chain clients
        for client in self.chain_clients.values():
            if hasattr(client, 'close'):
                await client.close()
        
        # Stop servers
        if self.web_server:
            await self.web_server.shutdown()
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        self.logger.info("Rainbow Bridge stopped")
    
    async def _initialize_chain_clients(self):
        """Initialize blockchain client connections"""
        try:
            # Solana client
            if ChainType.SOLANA in self.chains:
                self.chain_clients[ChainType.SOLANA] = SolanaClient(
                    self.chains[ChainType.SOLANA].rpc_url
                )
                self.logger.info("Connected to Solana network")
            
            # Stellar client
            if ChainType.STELLAR in self.chains:
                self.chain_clients[ChainType.STELLAR] = Server(
                    self.chains[ChainType.STELLAR].rpc_url
                )
                self.logger.info("Connected to Stellar network")
            
            # Tron client
            if ChainType.TRON in self.chains:
                self.chain_clients[ChainType.TRON] = tronpy.Tron(
                    HTTPProvider(self.chains[ChainType.TRON].rpc_url)
                )
                self.logger.info("Connected to Tron network")
            
            # Cardano (via CLI tools)
            if ChainType.CARDANO in self.chains:
                # Check if cardano-cli is available
                result = subprocess.run(['which', 'cardano-cli'], capture_output=True)
                if result.returncode == 0:
                    self.chain_clients[ChainType.CARDANO] = "cardano-cli"
                    self.logger.info("Cardano CLI tools available")
                else:
                    self.logger.warning("Cardano CLI tools not available")
            
            self.bridge_stats['chains_connected'] = len(self.chain_clients)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chain clients: {e}")
    
    async def _start_web_server(self):
        """Start HTTP API server"""
        try:
            app = web.Application()
            
            # Bridge API endpoints
            app.router.add_post('/api/bridge/create', self._handle_create_bridge)
            app.router.add_get('/api/bridge/{bridge_id}', self._handle_get_bridge)
            app.router.add_get('/api/bridge/status/{bridge_id}', self._handle_bridge_status)
            app.router.add_post('/api/bridge/confirm/{bridge_id}', self._handle_confirm_bridge)
            
            # Chain information
            app.router.add_get('/api/chains', self._handle_get_chains)
            app.router.add_get('/api/chains/{chain}/balance/{address}', self._handle_get_balance)
            
            # Statistics
            app.router.add_get('/api/stats', self._handle_get_stats)
            app.router.add_get('/api/transactions', self._handle_get_transactions)
            
            # Health check
            app.router.add_get('/health', self._handle_health)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', self.web_port)
            await site.start()
            
            self.web_server = app
            self.logger.info(f"HTTP API server started on port {self.web_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                '0.0.0.0',
                self.websocket_port
            )
            self.logger.info(f"WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _handle_create_bridge(self, request):
        """Create new bridge transaction"""
        try:
            data = await request.json()
            
            from_chain = ChainType(data['from_chain'])
            to_chain = ChainType(data['to_chain'])
            from_address = data['from_address']
            to_address = data['to_address']
            amount = Decimal(str(data['amount']))
            
            # Validate bridge parameters
            validation_result = await self._validate_bridge_request(
                from_chain, to_chain, from_address, to_address, amount
            )
            
            if not validation_result['valid']:
                return web.json_response({
                    'success': False,
                    'error': validation_result['error']
                }, status=400)
            
            # Create bridge transaction
            bridge = await self._create_bridge_transaction(
                from_chain, to_chain, from_address, to_address, amount
            )
            
            return web.json_response({
                'success': True,
                'bridge_id': bridge.bridge_id,
                'deposit_address': validation_result['deposit_address'],
                'amount': str(bridge.amount),
                'fee': str(bridge.fee),
                'estimated_time': validation_result['estimated_time']
            })
            
        except Exception as e:
            self.logger.error(f"Create bridge error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def _validate_bridge_request(self, from_chain: ChainType, to_chain: ChainType, 
                                     from_address: str, to_address: str, amount: Decimal) -> Dict:
        """Validate bridge transaction request"""
        try:
            # Check if chains are supported
            if from_chain not in self.chains or to_chain not in self.chains:
                return {'valid': False, 'error': 'Unsupported chain'}
            
            # Check amount limits
            from_config = self.chains[from_chain]
            if amount < from_config.min_amount:
                return {'valid': False, 'error': f'Amount below minimum {from_config.min_amount}'}
            
            if amount > from_config.max_amount:
                return {'valid': False, 'error': f'Amount above maximum {from_config.max_amount}'}
            
            # Validate addresses
            if not await self._validate_address(from_chain, from_address):
                return {'valid': False, 'error': 'Invalid from_address'}
            
            if not await self._validate_address(to_chain, to_address):
                return {'valid': False, 'error': 'Invalid to_address'}
            
            # Generate deposit address for this bridge
            deposit_address = await self._generate_deposit_address(from_chain, to_chain)
            
            # Calculate estimated completion time
            total_confirmations = from_config.confirmations_required + self.chains[to_chain].confirmations_required
            estimated_time = total_confirmations * 60  # Assume 1 minute per confirmation
            
            return {
                'valid': True,
                'deposit_address': deposit_address,
                'estimated_time': estimated_time
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    async def _validate_address(self, chain: ChainType, address: str) -> bool:
        """Validate address format for specific chain"""
        try:
            if chain == ChainType.ZION:
                return address.startswith('ZION') and len(address) >= 20
            
            elif chain == ChainType.SOLANA:
                # Solana addresses are 32-byte base58 encoded
                return len(address) >= 32 and len(address) <= 44
            
            elif chain == ChainType.STELLAR:
                # Stellar addresses start with G and are 56 characters
                return address.startswith('G') and len(address) == 56
            
            elif chain == ChainType.CARDANO:
                # Cardano addresses start with addr1
                return address.startswith('addr1') and len(address) >= 50
            
            elif chain == ChainType.TRON:
                # Tron addresses start with T and are 34 characters
                return address.startswith('T') and len(address) == 34
            
            return False
            
        except Exception:
            return False
    
    async def _generate_deposit_address(self, from_chain: ChainType, to_chain: ChainType) -> str:
        """Generate unique deposit address for bridge transaction"""
        try:
            # For demonstration, generate deterministic addresses based on chains
            seed = f"{from_chain.value}_{to_chain.value}_{time.time()}"
            hash_value = hashlib.sha256(seed.encode()).hexdigest()
            
            if from_chain == ChainType.ZION:
                return f"ZION_BRIDGE_{hash_value[:20].upper()}"
            elif from_chain == ChainType.SOLANA:
                return f"SOL_DEPOSIT_{hash_value[:16]}"
            elif from_chain == ChainType.STELLAR:
                return f"G{hash_value[:55].upper()}"
            elif from_chain == ChainType.CARDANO:
                return f"addr1_{hash_value[:48]}"
            elif from_chain == ChainType.TRON:
                return f"T{hash_value[:33]}"
            
            return f"BRIDGE_{hash_value[:20]}"
            
        except Exception as e:
            self.logger.error(f"Deposit address generation error: {e}")
            return "BRIDGE_ERROR"
    
    async def _create_bridge_transaction(self, from_chain: ChainType, to_chain: ChainType,
                                       from_address: str, to_address: str, amount: Decimal) -> BridgeTransaction:
        """Create new bridge transaction"""
        bridge_id = self._generate_bridge_id()
        fee = self.chains[from_chain].bridge_fee
        
        bridge = BridgeTransaction(
            bridge_id=bridge_id,
            from_chain=from_chain,
            to_chain=to_chain,
            from_address=from_address,
            to_address=to_address,
            from_txid="",  # Will be filled when deposit is detected
            to_txid=None,
            amount=amount,
            fee=fee,
            status=BridgeStatus.PENDING,
            created_at=int(time.time()),
            completed_at=None,
            confirmations=0,
            required_confirmations=self.chains[from_chain].confirmations_required
        )
        
        self.active_bridges[bridge_id] = bridge
        self.bridge_stats['total_bridges'] += 1
        
        self.logger.info(f"Created bridge {bridge_id[:8]} from {from_chain.value} to {to_chain.value}")
        return bridge
    
    async def _transaction_monitor_loop(self):
        """Monitor pending bridge transactions"""
        while self.running:
            try:
                for bridge_id, bridge in list(self.active_bridges.items()):
                    if bridge.status == BridgeStatus.PENDING:
                        # Check for deposit on source chain
                        if not bridge.from_txid:
                            deposit_txid = await self._check_for_deposit(bridge)
                            if deposit_txid:
                                bridge.from_txid = deposit_txid
                                self.logger.info(f"Deposit detected for bridge {bridge_id[:8]}: {deposit_txid[:16]}...")
                        
                        # Monitor confirmations
                        if bridge.from_txid:
                            confirmations = await self._get_transaction_confirmations(
                                bridge.from_chain, bridge.from_txid
                            )
                            bridge.confirmations = confirmations
                            
                            if confirmations >= bridge.required_confirmations:
                                # Execute bridge on destination chain
                                await self._execute_bridge_transfer(bridge)
                    
                    elif bridge.status == BridgeStatus.CONFIRMED and not bridge.to_txid:
                        # Monitor destination transaction
                        await self._monitor_destination_transaction(bridge)
                
            except Exception as e:
                self.logger.error(f"Transaction monitor error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _check_for_deposit(self, bridge: BridgeTransaction) -> Optional[str]:
        """Check for deposit transaction on source chain"""
        try:
            if bridge.from_chain == ChainType.ZION:
                return await self._check_zion_deposit(bridge)
            elif bridge.from_chain == ChainType.SOLANA:
                return await self._check_solana_deposit(bridge)
            elif bridge.from_chain == ChainType.STELLAR:
                return await self._check_stellar_deposit(bridge)
            elif bridge.from_chain == ChainType.CARDANO:
                return await self._check_cardano_deposit(bridge)
            elif bridge.from_chain == ChainType.TRON:
                return await self._check_tron_deposit(bridge)
            
        except Exception as e:
            self.logger.error(f"Deposit check error for {bridge.bridge_id[:8]}: {e}")
        
        return None
    
    async def _check_zion_deposit(self, bridge: BridgeTransaction) -> Optional[str]:
        """Check for ZION deposit transaction"""
        try:
            # Query ZION node for transactions to deposit address
            rpc_data = {
                'jsonrpc': '2.0',
                'method': 'gettransactions',
                'params': {'address': bridge.from_address, 'count': 10},
                'id': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.zion_rpc_url}/json_rpc", json=rpc_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        transactions = data.get('result', {}).get('transactions', [])
                        
                        for tx in transactions:
                            if (tx.get('amount', 0) >= bridge.amount and
                                tx.get('confirmations', 0) > 0):
                                return tx['txid']
            
        except Exception as e:
            self.logger.error(f"ZION deposit check error: {e}")
        
        return None
    
    async def _check_solana_deposit(self, bridge: BridgeTransaction) -> Optional[str]:
        """Check for Solana deposit transaction"""
        try:
            client = self.chain_clients.get(ChainType.SOLANA)
            if not client:
                return None
            
            # Get recent transactions for the deposit address
            # This is a simplified implementation
            self.logger.debug(f"Checking Solana deposits for bridge {bridge.bridge_id[:8]}")
            
            # TODO: Implement actual Solana transaction monitoring
            return None
            
        except Exception as e:
            self.logger.error(f"Solana deposit check error: {e}")
            return None
    
    async def _execute_bridge_transfer(self, bridge: BridgeTransaction):
        """Execute transfer on destination chain"""
        try:
            bridge.status = BridgeStatus.CONFIRMED
            
            if bridge.to_chain == ChainType.ZION:
                txid = await self._execute_zion_transfer(bridge)
            elif bridge.to_chain == ChainType.SOLANA:
                txid = await self._execute_solana_transfer(bridge)
            elif bridge.to_chain == ChainType.STELLAR:
                txid = await self._execute_stellar_transfer(bridge)
            elif bridge.to_chain == ChainType.CARDANO:
                txid = await self._execute_cardano_transfer(bridge)
            elif bridge.to_chain == ChainType.TRON:
                txid = await self._execute_tron_transfer(bridge)
            else:
                raise Exception(f"Unsupported destination chain: {bridge.to_chain}")
            
            if txid:
                bridge.to_txid = txid
                bridge.completed_at = int(time.time())
                self.bridge_stats['successful_bridges'] += 1
                self.bridge_stats['total_volume'] += bridge.amount
                
                self.logger.info(f"Bridge {bridge.bridge_id[:8]} completed: {txid[:16]}...")
                
                # Move to completed bridges
                self.completed_bridges.append(bridge)
                if bridge.bridge_id in self.active_bridges:
                    del self.active_bridges[bridge.bridge_id]
            else:
                bridge.status = BridgeStatus.FAILED
                self.bridge_stats['failed_bridges'] += 1
                
        except Exception as e:
            self.logger.error(f"Bridge transfer execution error: {e}")
            bridge.status = BridgeStatus.FAILED
            self.bridge_stats['failed_bridges'] += 1
    
    async def _execute_zion_transfer(self, bridge: BridgeTransaction) -> Optional[str]:
        """Execute ZION transfer"""
        try:
            # Create ZION transaction
            rpc_data = {
                'jsonrpc': '2.0',
                'method': 'sendtransaction',
                'params': {
                    'to_address': bridge.to_address,
                    'amount': int(bridge.amount * 1000000),  # Convert to atomic units
                    'fee': int(bridge.fee * 1000000)
                },
                'id': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.zion_rpc_url}/json_rpc", json=rpc_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('result', {}).get('txid')
            
        except Exception as e:
            self.logger.error(f"ZION transfer error: {e}")
        
        return None
    
    def _generate_bridge_id(self) -> str:
        """Generate unique bridge transaction ID"""
        return hashlib.sha256(f"bridge_{time.time()}_{self.bridge_private_key[:8]}".encode()).hexdigest()
    
    async def _handle_get_chains(self, request):
        """Get supported chains information"""
        chains_info = {}
        
        for chain_type, config in self.chains.items():
            chains_info[chain_type.value] = {
                'name': config.name,
                'network_id': config.network_id,
                'min_amount': str(config.min_amount),
                'max_amount': str(config.max_amount),
                'bridge_fee': str(config.bridge_fee),
                'confirmations_required': config.confirmations_required,
                'connected': chain_type in self.chain_clients
            }
        
        return web.json_response({
            'success': True,
            'chains': chains_info
        })
    
    async def _handle_get_stats(self, request):
        """Get bridge statistics"""
        return web.json_response({
            'success': True,
            'stats': {
                **self.bridge_stats,
                'total_volume': str(self.bridge_stats['total_volume']),
                'active_bridges': len(self.active_bridges),
                'completed_bridges': len(self.completed_bridges)
            }
        })
    
    async def _handle_health(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'version': '2.6.75',
            'uptime': time.time() - getattr(self, 'start_time', time.time()),
            'chains_connected': self.bridge_stats['chains_connected']
        })


# CLI interface for running rainbow bridge
async def run_rainbow_bridge():
    """Run ZION Rainbow Bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ZION Rainbow Bridge')
    parser.add_argument('--private-key', required=True, help='Bridge private key')
    parser.add_argument('--zion-rpc', default='http://localhost:18089', help='ZION node RPC URL')
    parser.add_argument('--web-port', type=int, default=9000, help='Web API port')
    parser.add_argument('--ws-port', type=int, default=9001, help='WebSocket port')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start rainbow bridge
    bridge = ZionRainbowBridge(
        bridge_private_key=args.private_key,
        zion_rpc_url=args.zion_rpc,
        web_port=args.web_port,
        websocket_port=args.ws_port
    )
    
    try:
        print(f"🌈 Starting ZION Rainbow Bridge...")
        print(f"   API Port: {args.web_port}")
        print(f"   WebSocket Port: {args.ws_port}")
        print("   Supported chains: ZION, Solana, Stellar, Cardano, Tron")
        print("   Press Ctrl+C to stop")
        
        bridge.start_time = time.time()
        await bridge.start()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down rainbow bridge...")
        await bridge.stop()
        print("✅ Rainbow bridge stopped")


if __name__ == "__main__":
    asyncio.run(run_rainbow_bridge())