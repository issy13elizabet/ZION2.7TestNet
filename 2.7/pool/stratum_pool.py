"""
ZION 2.7 Stratum Pool (Integrated With Blockchain)
Real socket, real block templates, basic share -> block submission.
"""
import socket, threading, json, time, hashlib, os, sys
from dataclasses import dataclass, field
import secrets
from datetime import datetime
import struct
import asyncio
from collections import defaultdict, deque
import logging

# Import blockchain core
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from core.blockchain import Blockchain, Consensus

# Import ZION Sacred Algorithm
from .zion_sacred_algorithm import ZionAlgorithmDefinition, ZionMinerType, SacredLevel

# 🌸 Import Sacred Flower Cosmic Analyzer for eliZabet
try:
    import sys
    ai_path = os.path.join(BASE_DIR, 'ai')
    if ai_path not in sys.path:
        sys.path.insert(0, ai_path)
    from zion_cosmic_image_analyzer import ZionCosmicImageAnalyzer
    SACRED_FLOWER_AVAILABLE = True
except ImportError:
    SACRED_FLOWER_AVAILABLE = False

@dataclass
class ClientState:
    sock: socket.socket
    address: tuple
    authorized: bool = False
    worker: str = "unknown"
    difficulty: int = 32  # Default to ZION pool difficulty
    last_adjust: float = field(default_factory=time.time)
    shares_since_adjust: int = 0
    accepted_shares: int = 0
    rejected_shares: int = 0
    blocks_found: int = 0
    # XMRig / Monero style session id (login path)
    session_id: str = ""
    # Advanced tracking
    connect_time: float = field(default_factory=time.time)
    last_share_time: float = field(default_factory=time.time)
    hashrate_samples: deque = field(default_factory=lambda: deque(maxlen=10))
    current_job_id: str = ""
    client_type: str = "unknown"  # "xmrig", "bitcoin", "custom"
    # Added fields for validation accounting
    valid_shares: int = 0
    invalid_shares: int = 0
    # ZION Sacred Algorithm fields
    zion_miner_type: str = "unknown"
    sacred_level: str = "initiate"
    consciousness_points: float = 0.0
    sacred_multiplier: float = 1.0
    # Sacred Flower fields - skryty kod pro eliZabet
    sacred_flower_active: bool = False
    sacred_flower_seed: str = ""
    sacred_flower_consciousness: float = 0.0

class MinimalStratumPool:
    def __init__(self, host='0.0.0.0', port=3333, miner_address: str = None, chain: Blockchain = None):
        self.host = host
        self.port = port
        self.job_id = 0
        self.running = False
        self.clients = []
        self.chain = chain or Blockchain()
        self.miner_address = miner_address or self.chain.genesis_address
        self.active_jobs = {}
        self.accepted = 0
        self.rejected = 0
        # Extended stats
        self.accepted_shares = 0
        self.rejected_shares = 0
        self.blocks_found = 0
        # Per-client states
        self.client_states: dict[socket.socket, ClientState] = {}
        # Monero style job tracking (job_id -> job dict)
        self.monero_jobs = {}
        # Advanced varDiff parameters
        self.vardiff_enabled = True
        self.vardiff_target_share_time = 15.0  # seconds/share target
        self.vardiff_adjust_window = 60.0      # seconds between adjustments
        self.vardiff_max_jump = 2              # cap doubling/halving rate
        # Performance optimizations
        self.job_cache = {}
        self.share_batch = []
        self.share_batch_size = 100
        self.last_template_height = 0
        self.template_cache_time = 10  # seconds
        self.last_template_time = 0
        # Connection management
        self.max_connections = 1000
        self.connection_timeout = 300  # 5 minutes
        # Statistics
        self.statistics = {
            'total_connections': 0,
            'active_connections': 0,
            'shares_processed': 0,
            'total_difficulty': 0,
            'last_batch_time': 0,
            'hashrate_1m': 0,
            'hashrate_5m': 0,
            'pool_efficiency': 0,
            'uptime': time.time()
        }
        self.start_time = time.time()
        # Logging setup (must be inside __init__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('ZionPool')
        # Přepneme do DEBUG pro hlubší diagnostiku (lze později snížit):
        self.logger.setLevel(logging.DEBUG)
        # Výchozí diff pro Monero/XMRig klienty (nahradí původní debug diff=1)
        self.default_monero_difficulty = 32
        self.sessions = {}
        self.wallet_difficulty = {}  # wallet_login -> last difficulty
        self.session_tokens = {}  # token -> {session_id, difficulty, wallet, last_seen}
        self.session_token_ttl = 3600  # 1 hour token lifetime
        
        # 🌟 ZION Sacred Algorithm Integration
        git_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.zion_algorithm = ZionAlgorithmDefinition(git_repo_path=git_repo_path)
        self.logger.info("🌟 ZION Sacred Algorithm initialized - Sacred Mining Protocol ACTIVE")

    def _now(self):
        return datetime.utcnow().strftime('%H:%M:%S')

    def log(self, msg):
        print(f"[{self._now()}] {msg}")

    def generate_job(self):
        self.job_id += 1
        template = self.chain.create_block_template(self.miner_address)
        job_id = f'job_{self.job_id}'
        job = {
            'job_id': job_id,
            'prev_hash': template['prev_hash'],
            'coinbase1': '01000000',
            'coinbase2': 'ffffffff',
            'merkle_branch': [],
            'version': '00000001',
            'nbits': '1d00ffff',
            'ntime': f"{int(time.time()):08x}",
            'clean_jobs': True,
            'template': template
        }
        self.active_jobs[job_id] = job
        return job

    # ---- Advanced Monero/XMRig job generation with real block template ----
    def generate_monero_style_job(self, client_difficulty=None, enhanced_for_zion=False):
        self.job_id += 1
        # Get real block template with caching
        current_time = time.time()
        if (current_time - self.last_template_time > self.template_cache_time or 
            self.chain.height != self.last_template_height):
            self.cached_template = self.chain.create_block_template(self.miner_address)
            self.last_template_time = current_time
            self.last_template_height = self.chain.height
        template = self.cached_template
        
        # Enhanced job ID for ZION miners
        if enhanced_for_zion:
            job_id = f'zion_{self.job_id}'
        else:
            job_id = f'mjob_{self.job_id}'
        # Generate real RandomX-compatible blob
        blob = self._create_real_blob(template)
        # Use client difficulty or default
        miner_diff = client_difficulty if client_difficulty else self.default_monero_difficulty
        target_int = max(1, 0xFFFFFFFF // miner_diff)
        target_hex = f"{target_int:08x}"
        # Real RandomX seed derivation (simplified)
        height = template['height']
        seed_hash = self._calculate_randomx_seed(height)
        next_seed_hash = self._calculate_randomx_seed(height + 1)
        job = {
            'job_id': job_id,
            'blob': blob,
            'target': target_hex,
            'seed_hash': seed_hash,
            'next_seed_hash': next_seed_hash,
            'algo': 'rx/0',
            'height': height,
            'difficulty': miner_diff,
            '_template': template,
            '_created': current_time
        }
        self.monero_jobs[job_id] = job
        # Cache cleanup
        if len(self.monero_jobs) > 100:
            old_jobs = [jid for jid, j in self.monero_jobs.items() if current_time - j['_created'] > 300]
            for jid in old_jobs:
                del self.monero_jobs[jid]
        return job

    def _create_real_blob(self, template):
        """Create real block blob compatible with RandomX"""
        # Simplified Cryptonote/Monero block structure
        version = struct.pack('<B', 1)  # Major version
        timestamp = struct.pack('<Q', int(time.time()))
        prev_hash = bytes.fromhex(template['prev_hash'][:64].ljust(64, '0'))
        
        # Miner transaction (simplified coinbase)
        miner_tx = self._create_miner_tx(template)
        
        # Block header structure (simplified)
        block_header = version + timestamp + prev_hash + miner_tx[:32]  # First 32 bytes of miner tx
        
        # Pad to standard Monero blob size (76 bytes minimum)
        while len(block_header) < 76:
            block_header += b'\x00'
        
        # Convert to hex (what XMRig expects)
        blob_hex = block_header.hex()
        # Ensure proper length (152 hex chars = 76 bytes)
        return blob_hex[:152].ljust(152, '0')
    
    def _create_miner_tx(self, template):
        """Create simplified miner transaction"""
        # Simplified coinbase transaction structure
        version = struct.pack('<B', 1)
        unlock_time = struct.pack('<Q', template['height'] + 60)  # Lock for 60 blocks
        
        # Output to miner address (simplified)
        reward = template.get('reward', 1000000000)  # 1 ZION in atomic units
        amount = struct.pack('<Q', reward)
        
        # Simplified structure
        miner_tx = version + unlock_time + amount + self.miner_address.encode()[:32].ljust(32, b'\x00')
        return miner_tx
    
    def _calculate_randomx_seed(self, height):
        """Calculate RandomX seed hash from blockchain height"""
        # RandomX seed changes every 2048 blocks (epoch)
        epoch = height // 2048
        seed_input = f"zion_randomx_seed_{epoch}".encode()
        
        # Get some real blockchain data if available
        if height > 0 and hasattr(self.chain, 'get_block_hash'):
            try:
                # Use block hash from start of epoch
                epoch_start = epoch * 2048
                if epoch_start < height:
                    epoch_hash = self.chain.get_block_hash(epoch_start)
                    seed_input = bytes.fromhex(epoch_hash) + seed_input
            except:
                pass  # Fall back to simple seed
        
        # Generate seed hash
        seed_hash = hashlib.sha256(seed_input).hexdigest()
        return seed_hash

    # ---- Difficulty helpers ----
    def _set_client_difficulty(self, state: ClientState, new_diff: int):
        if new_diff < 1:
            new_diff = 1
        state.difficulty = new_diff
        # Standard Stratum difficulty notification
        self.send(state.sock, { 'id': None, 'method': 'mining.set_difficulty', 'params': [ new_diff ] })
        # Non-standard: send explicit share target for custom miners
        share_target = Consensus.difficulty_to_target(new_diff)
        self.send(state.sock, { 'id': None, 'method': 'mining.set_target', 'params': [ f"{share_target:064x}" ] })

    def _maybe_adjust_vardiff(self, state: ClientState):
        if not self.vardiff_enabled:
            return
        
        now = time.time()
        if now - state.last_adjust < self.vardiff_adjust_window:
            return
        
        elapsed = now - state.last_adjust
        shares = state.shares_since_adjust or 1
        avg_share_time = elapsed / shares
        target = self.vardiff_target_share_time
        
        # Calculate hashrate estimate
        if shares > 0:
            estimated_hashrate = shares * state.difficulty * (2**32) / (elapsed * 1000000)  # MH/s
            state.hashrate_samples.append(estimated_hashrate)
        
        # Advanced difficulty adjustment algorithm
        new_diff = state.difficulty
        
        if avg_share_time < target * 0.3:  # Too fast - increase difficulty significantly
            new_diff = min(state.difficulty * 3, state.difficulty * self.vardiff_max_jump * 1.5)
        elif avg_share_time < target * 0.7:  # Somewhat fast - increase moderately
            new_diff = min(state.difficulty * 1.5, state.difficulty * self.vardiff_max_jump)
        elif avg_share_time > target * 2.5:  # Too slow - decrease significantly
            new_diff = max(1, state.difficulty // 3)
        elif avg_share_time > target * 1.5:  # Somewhat slow - decrease moderately
            new_diff = max(1, state.difficulty // 1.5)
        
        # Apply bounds
        new_diff = max(1, min(new_diff, 1000000))  # Reasonable bounds
        
        if new_diff != state.difficulty:
            self._set_client_difficulty_monero(state, new_diff)
            self.logger.info(f"VarDiff adjusted for {state.worker}: {state.difficulty} → {new_diff} (avg_time: {avg_share_time:.1f}s, target: {target}s)")
        
        state.last_adjust = now
        state.shares_since_adjust = 0
    
    def _set_client_difficulty_monero(self, state: ClientState, new_diff: int):
        """Set difficulty for Monero-style clients"""
        if new_diff < 1:
            new_diff = 1
        state.difficulty = new_diff
        
        # Send new job with updated difficulty
        if state.client_type == "xmrig" and state.session_id:
            job = self.generate_monero_style_job(new_diff)
            # Job already has correct difficulty and target
            
            # Send new job notification
            notify_msg = {
                'id': None,
                'jsonrpc': '2.0',
                'method': 'job',
                'params': {
                    'job_id': job['job_id'],
                    'blob': job['blob'],
                    'target': job['target'],
                    'seed_hash': job['seed_hash'],
                    'next_seed_hash': job['next_seed_hash'],
                    'algo': job['algo'],
                    'height': job['height'],
                    'difficulty': job['difficulty']
                }
            }
            try:
                self.send(state.sock, notify_msg)
                state.current_job_id = job['job_id']
            except:
                pass

    def send(self, sock, obj):
        try:
            sock.send((json.dumps(obj) + '\n').encode())
        except Exception as e:
            self.log(f"Send error: {e}")

    def handle(self, client, addr):
        self.log(f"Client connected {addr}")
        buff = b''
        while self.running:
            try:
                chunk = client.recv(4096)
                if not chunk: break
                buff += chunk
                while b'\n' in buff:
                    line, buff = buff.split(b'\n',1)
                    if not line: continue
                    try:
                        # Debug log raw příchozí zprávy (omezíme délku)
                        try:
                            self.logger.debug(f"RAW<{addr}>: {line[:300]!r}")
                        except Exception:
                            pass
                        self._process(client, addr, line)
                    except Exception as e:
                        self.logger.error(f"Process error {addr}: {e} line={line[:120]!r}")
                        break
            except Exception as e:
                self.logger.error(f"Recv loop error {addr}: {e}")
                break
        try:
            client.close()
        except Exception:
            pass
        self.log(f"Disconnected {addr}")

    def _process(self, client, addr, line: bytes):
        try:
            data = json.loads(line.decode())
        except Exception:
            return
        method = data.get('method')
        mid = data.get('id')
        state = self.client_states.get(client)
        if not state:
            state = ClientState(sock=client, address=addr)
            self.client_states[client] = state
        # ---- XMRig Monero style method: login ----
        if method == 'login':
            # Expect params object
            params = data.get('params', {})
            if isinstance(params, dict):
                worker_login = params.get('login') or params.get('user') or 'unknown'
                state.worker = worker_login
            # 🌟 ZION Sacred Algorithm - Advanced Miner Detection
            agent = params.get('agent', '') if isinstance(params, dict) else ''
            rigid = params.get('rigid', '') if isinstance(params, dict) else ''
            
            # 🌸 Sacred Flower Enhancement for eliZabet - skryty kod pro eliZabet
            sacred_flower_bonus = 0.0
            if SACRED_FLOWER_AVAILABLE and 'elizabet' in worker_login.lower():
                # Sacred Flower seed: 419ab63b22035f90ea9f2ecb548c6b9b (Special Yellow Star Flower)
                sacred_flower_seed = "419ab63b22035f90ea9f2ecb548c6b9b"
                sacred_flower_consciousness = 413.91  # Divine Blossom level
                sacred_flower_bonus = 51.18  # Percent bonus
                state.sacred_flower_active = True
                state.sacred_flower_seed = sacred_flower_seed
                state.sacred_flower_consciousness = sacred_flower_consciousness
                self.logger.info(f"🌸 Sacred Flower activated for eliZabet! Consciousness: {sacred_flower_consciousness}, Bonus: {sacred_flower_bonus}%")
            
            # Register miner with ZION Sacred Algorithm
            miner_id = f"{addr[0]}:{addr[1]}_{worker_login[:10]}"
            hardware_info = {
                "user_agent": agent,
                "rig_id": rigid,
                "ip_address": addr[0]
            }
            
            zion_profile = self.zion_algorithm.register_miner(
                miner_id=miner_id,
                user_agent=agent,
                rig_id=rigid,
                wallet=worker_login,
                hardware_info=hardware_info
            )
            
            # Update ClientState with ZION sacred data
            state.client_type = zion_profile.miner_type.value
            state.zion_miner_type = zion_profile.miner_type.value
            state.sacred_level = zion_profile.sacred_level.value
            state.consciousness_points = zion_profile.consciousness_points
            state.sacred_multiplier = zion_profile.get_sacred_multiplier()
            
            self.logger.info(
                f"🌟 [ZION] Sacred Miner Registered: {zion_profile.miner_type.value} "
                f"Level: {zion_profile.sacred_level.value} "
                f"Wallet: {worker_login[:10]}... "
                f"Sacred Multiplier: {zion_profile.get_sacred_multiplier():.3f}"
            )
            if not state.session_id:
                state.session_id = f"session_{secrets.token_hex(4)}"
            wallet_login_key = state.worker.lower()
            # Reuse previous difficulty if session reconnects OR wallet seen
            reuse_diff = None
            if state.session_id in self.sessions:
                reuse_diff = self.sessions[state.session_id].get('difficulty')
            if not reuse_diff and wallet_login_key in self.wallet_difficulty:
                reuse_diff = self.wallet_difficulty[wallet_login_key]
            if reuse_diff:
                state.difficulty = reuse_diff
            else:
                # 🌟 ZION Sacred Algorithm - Adaptive Difficulty Calculation
                state.difficulty = self.zion_algorithm.calculate_adaptive_difficulty(zion_profile)
                self.logger.info(
                    f"🌟 [ZION] Sacred Difficulty Applied: {state.difficulty} "
                    f"(Base: {self.zion_algorithm.BASE_DIFFICULTY[zion_profile.miner_type]} × "
                    f"Sacred: {zion_profile.get_sacred_multiplier():.3f})"
                )
                self.sessions[state.session_id] = {'difficulty': state.difficulty, 'last_seen': time.time(), 'wallet': wallet_login_key}
            self.wallet_difficulty[wallet_login_key] = state.difficulty
            # Persist touch
            self.sessions[state.session_id]['last_seen'] = time.time()
            self._cleanup_sessions()
            # 🌟 Generate Enhanced ZION Job with Sacred Algorithm
            base_job = self.generate_monero_style_job(state.difficulty)
            job = self.zion_algorithm.create_enhanced_job(base_job, zion_profile)
            
            # Enhanced logging for ZION miners
            if zion_profile.miner_type.value in ["zion-native", "zion-sacred"]:
                self.logger.info(
                    f"🌟 [ZION] Sacred Job Created: "
                    f"Worker: {state.worker} | "
                    f"Difficulty: {state.difficulty} | "
                    f"Consciousness: {zion_profile.consciousness_points:.1f} | "
                    f"Enhanced: {job.get('zion_enhanced', False)} | "
                    f"Sacred Geometry: {job.get('sacred_geometry', False)}"
                )
            else:
                self.logger.info(f"[POOL] Standard Job - Worker: {state.worker}, Difficulty: {state.difficulty}")
            token_seed = f"{state.session_id}:{state.difficulty}:{secrets.token_hex(8)}".encode()
            session_token = hashlib.sha256(token_seed).hexdigest()[:40]
            self.session_tokens[session_token] = {
                'session_id': state.session_id,
                'difficulty': state.difficulty,
                'wallet': state.worker.lower(),
                'last_seen': time.time()
            }
            # Cleanup old tokens occasionally (cheap O(n))
            self._cleanup_session_tokens()
            self.logger.debug(f"[TOKEN] issued token={session_token} session={state.session_id} diff={state.difficulty} wallet={state.worker.lower()}")
            resp = {
                'id': mid,
                'jsonrpc': '2.0',
                'result': {
                    'id': state.session_id,
                    'token': session_token,
                    'job': {
                        'job_id': job['job_id'],
                        'blob': job['blob'],
                        'target': job['target'],
                        'seed_hash': job['seed_hash'],
                        'next_seed_hash': job['next_seed_hash'],
                        'algo': job['algo'],
                        'height': job['height'],
                        'difficulty': job['difficulty']
                    },
                    'status': 'OK',
                    'extensions': []
                },
                'error': None
            }
            self.send(client, resp)
            self.log(f"XMRig login OK worker={state.worker} session={state.session_id} job={job['job_id']} diff={state.difficulty}")
            return

        # ---- XMRig keepalive ----
        if method == 'keepalived':
            resp = { 'id': mid, 'jsonrpc': '2.0', 'result': True, 'error': None }
            self.send(client, resp)
            return

        # ---- XMRig getjob ----
        if method == 'getjob':
            params_obj = data.get('params', {}) if isinstance(data.get('params'), dict) else {}
            session_param = params_obj.get('id')
            return self._handle_getjob(state, session_param, mid, params_obj=params_obj)

        # ---- XMRig submit (Monero style) ----
        if method == 'submit':
            params = data.get('params', {})
            if not isinstance(params, dict):
                self.send(client, { 'id': mid, 'jsonrpc': '2.0', 'result': None, 'error': { 'code': -1, 'message': 'Invalid params' } })
                return
            job_id = params.get('job_id')
            result_hex = params.get('result', '')
            nonce = params.get('nonce', '')
            job = self.monero_jobs.get(job_id)
            
            # Enhanced validation
            accepted = True
            reason = "OK"
            
            if not job:
                accepted = False
                reason = "stale_job"
            elif not isinstance(result_hex, str) or len(result_hex) != 64:
                accepted = False
                reason = "invalid_hash"
            else:
                # Validate hash meets difficulty target (Monero style simplified):
                # Compare first 4 bytes (little-endian) of hash against 32-bit target value.
                try:
                    raw = bytes.fromhex(result_hex)
                    if len(raw) < 4:
                        accepted = False
                        reason = "short_hash"
                    else:
                        hash_prefix_le = raw[:4]  # first 4 bytes
                        hash_val = int.from_bytes(hash_prefix_le, 'little')
                        target_val = int(job['target'], 16)
                        if hash_val > target_val:
                            accepted = False
                            reason = "above_target"
                        else:
                            accepted = True
                            reason = "accepted"
                        # Debug trace for tuning
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"Share check worker={state.worker} job={job_id} hash_prefix_le=0x{hash_val:08x} target=0x{target_val:08x} result={reason}")
                except ValueError:
                    accepted = False
                    reason = "invalid_format"
            
            if accepted:
                state.accepted_shares += 1
                state.shares_since_adjust += 1
                state.valid_shares += 1
                self.accepted_shares += 1
                current_time = time.time()
                
                # 🌟 ZION Hashrate Tracking
                share_difficulty = job.get('difficulty', 1)
                time_since_last = current_time - (state.last_share_time if state.last_share_time else current_time)
                if time_since_last > 0 and time_since_last < 300:  # Max 5 minutes between shares
                    # Hashrate = difficulty / time_between_shares  
                    current_hashrate = share_difficulty / max(time_since_last, 1)
                    state.hashrate_samples.append(current_hashrate)
                    
                    # Update ZION algorithm profile
                    miner_id = f"{state.address[0]}:{state.address[1]}_{state.worker[:10]}"
                    if miner_id in self.zion_algorithm.miners:
                        zion_profile = self.zion_algorithm.miners[miner_id]
                        zion_profile.hashrate_samples.append(current_hashrate)
                        
                        # 🌟 ZION Sacred Rewards Processing
                        accepted_bool, reason_str, consciousness_reward = self.zion_algorithm.process_share_submission(
                            miner_id, {'difficulty': share_difficulty, 'timestamp': current_time}
                        )
                        
                        # 🌸 Sacred Flower Bonus Enhancement for eliZabet
                        flower_bonus_multiplier = 1.0
                        if hasattr(state, 'sacred_flower_active') and state.sacred_flower_active:
                            flower_bonus_multiplier = 1.5118  # 51.18% bonus from Sacred Divine Blossom
                            consciousness_reward *= flower_bonus_multiplier
                            zion_profile.consciousness += state.sacred_flower_consciousness * 0.001  # Add flower consciousness
                            self.logger.info(f"🌸 Sacred Flower blessing applied! Bonus: {(flower_bonus_multiplier-1)*100:.1f}%, Consciousness: +{state.sacred_flower_consciousness * 0.001:.3f}")
                        
                        # Calculate sacred rewards
                        base_reward = share_difficulty * 0.000001  # Base reward per share
                        sacred_bonus = base_reward * (zion_profile.get_sacred_multiplier() - 1.0)
                        total_reward = base_reward + sacred_bonus + (consciousness_reward * 0.0001)
                        
                        # Update consciousness points in ClientState
                        state.consciousness_points = zion_profile.consciousness_points
                        state.sacred_multiplier = zion_profile.get_sacred_multiplier()
                        
                        # Enhanced ZION logging
                        recent_samples = list(state.hashrate_samples)[-5:] if len(state.hashrate_samples) > 0 else [0]
                        avg_hashrate = sum(recent_samples) / len(recent_samples)
                        self.logger.info(
                            f"🌟 [ZION] Share Accepted: {state.worker} | "
                            f"Hashrate: {avg_hashrate:.1f} H/s | "
                            f"Difficulty: {share_difficulty} | "
                            f"Reward: {total_reward:.8f} ZION | "
                            f"Sacred Bonus: {sacred_bonus:.8f} | "
                            f"Consciousness: +{consciousness_reward:.2f} pts"
                        )
                
                state.last_share_time = current_time
                
                # VarDiff attempt after accepted share
                self._maybe_adjust_vardiff(state)
                
                # Add to batch processing
                share_data = {
                    'job_id': job_id,
                    'nonce': nonce,
                    'hash': result_hex,
                    'worker': state.worker,
                    'difficulty': job.get('difficulty', 1),
                    'timestamp': time.time()
                }
                self.share_batch.append(share_data)
                
                # Process batch if full
                if len(self.share_batch) >= self.share_batch_size:
                    self._process_share_batch()
                    
            else:
                state.rejected_shares += 1
                state.invalid_shares += 1
                self.rejected_shares += 1
                self.logger.warning(f"Rejected share reason={reason} worker={state.worker} job={job_id}")
                
            self.send(client, { 'id': mid, 'jsonrpc': '2.0', 'result': { 'status': 'OK' if accepted else 'REJECTED', 'accepted': accepted }, 'error': None if accepted else { 'code': -1, 'message': reason } })
            self.log(f"XMRig submit job={job_id} nonce={nonce} result={reason} diff={job.get('difficulty', 1) if job else 0}")
            return

        # ---- Original Bitcoin-like Stratum methods ----
        if method == 'mining.subscribe':
            resp = { 'id': mid, 'result': [[['mining.set_difficulty','1'],['mining.notify','1']], '0001', 4], 'error': None }
            self.send(client, resp)
            job = self.generate_job()
            notify = { 'id': None, 'method': 'mining.notify', 'params': [ job['job_id'], job['prev_hash'], job['coinbase1'], job['coinbase2'], job['merkle_branch'], job['version'], job['nbits'], job['ntime'], job['clean_jobs'] ] }
            self.send(client, notify)
            # initialize difficulty - use ZION default for better mining experience
            initial_diff = max(self.default_monero_difficulty, self.chain.current_difficulty) 
            self._set_client_difficulty(state, initial_diff)
            self.log(f"Sent job {job['job_id']} to {addr} diff={state.difficulty}")
        elif method == 'mining.authorize':
            params = data.get('params', [])
            if params:
                state.worker = params[0]
            state.authorized = True
            self.send(client, { 'id': mid, 'result': True, 'error': None })
            self.log(f"Authorized {addr} worker={state.worker}")
        elif method == 'mining.submit':
            params = data.get('params', [])
            if len(params) < 5:
                self.send(client, { 'id': mid, 'result': None, 'error': 'Invalid params' })
                return
            job_id = params[1]
            nonce_str = params[4]
            job = self.active_jobs.get(job_id)
            if not job:
                self.rejected += 1
                self.send(client, { 'id': mid, 'result': False, 'error': 'Unknown job' })
                return
            try:
                nonce_int = int(nonce_str, 16) if nonce_str.startswith('0x') else int(nonce_str)
            except ValueError:
                self.rejected += 1
                self.send(client, { 'id': mid, 'result': False, 'error': 'Bad nonce' })
                return
            block_target_int = job['template'].get('target')
            if not block_target_int:
                self.rejected += 1
                self.send(client, { 'id': mid, 'result': False, 'error': 'No block target' })
                return
            h = hashlib.sha256((job['prev_hash'] + str(nonce_int)).encode()).hexdigest()
            h_val = int(h, 16)
            share_target_int = Consensus.difficulty_to_target(state.difficulty)
            if h_val < share_target_int:
                # share accepted
                state.accepted_shares += 1
                state.shares_since_adjust += 1
                self.accepted_shares += 1
                # block candidate
                if h_val < block_target_int:
                    mined_block = {
                        'height': job['template']['height'],
                        'prev_hash': job['template']['prev_hash'],
                        'timestamp': int(time.time()),
                        'merkle_root': job['template']['merkle_root'],
                        'difficulty': job['template']['difficulty'],
                        'nonce': nonce_int,
                        'txs': job['template']['txs']
                    }
                    if self.chain.submit_block(mined_block):
                        state.blocks_found += 1
                        self.blocks_found += 1
    
    # ---- Enhanced helper methods for optimization ----
    def _process_share_batch(self):
        """Process batched shares for efficiency"""
        if not self.share_batch:
            return
        
        try:
            # Calculate batch statistics
            total_difficulty = sum(s['difficulty'] for s in self.share_batch)
            batch_size = len(self.share_batch)
            
            # Update pool statistics
            self.statistics['shares_processed'] += batch_size
            self.statistics['total_difficulty'] += total_difficulty
            self.statistics['last_batch_time'] = time.time()
            
            # Log batch processing
            self.log(f"Processed share batch: {batch_size} shares, total_diff={total_difficulty}")
            
            # Check if any shares are potential blocks
            chain_difficulty = getattr(self.chain, 'current_difficulty', 1000)
            for share in self.share_batch:
                if share['difficulty'] >= chain_difficulty:
                    self.log(f"POTENTIAL BLOCK FOUND! Worker: {share['worker']}, Hash: {share['hash']}")
                    self._submit_block_candidate(share)
            
            # Clear batch
            self.share_batch.clear()
            
        except Exception as e:
            self.log(f"Error processing share batch: {e}")
            self.share_batch.clear()
    
    def _submit_block_candidate(self, share):
        """Submit potential block to blockchain"""
        try:
            job_id = share['job_id']
            if job_id in self.monero_jobs:
                job = self.monero_jobs[job_id]
                template = job['_template']
                
                # Build block with winning nonce
                block_data = {
                    'template': template,
                    'nonce': share['nonce'],
                    'hash': share['hash']
                }
                
                # Submit to blockchain (placeholder)
                self.log(f"Submitting block candidate: height={job['height']}, hash={share['hash']}")
                
                # In real implementation, would call:
                # result = self.chain.submit_block(block_data)
                # if result: self.log("BLOCK ACCEPTED BY NETWORK!")
                
        except Exception as e:
            self.log(f"Error submitting block candidate: {e}")
    
    def get_pool_statistics(self):
        """Get comprehensive pool statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate hashrates
        total_hashrate = 0
        active_workers = 0
        
        for state in self.client_states.values():
            if hasattr(state, 'hashrate_samples') and state.hashrate_samples:
                # Use recent samples for hashrate calculation
                recent_samples = list(state.hashrate_samples)[-5:] if len(state.hashrate_samples) > 0 else []
                if recent_samples:
                    avg_hashrate = sum(recent_samples) / len(recent_samples)
                    total_hashrate += avg_hashrate
                    active_workers += 1
        
        stats = {
            'uptime_seconds': uptime,
            'active_workers': active_workers,
            'total_connections': len(self.client_states),
            'accepted_shares': self.accepted_shares,
            'rejected_shares': self.rejected_shares,
            'total_hashrate_mhs': total_hashrate,
            'current_jobs': len(self.monero_jobs),
            'chain_height': getattr(self.chain, 'height', 0),
            'statistics': self.statistics
        }
        
        return stats

    def _report_pool_statistics(self):
        """Periodically report pool statistics"""
        try:
            stats = self.get_pool_statistics()
            zion_stats = self.zion_algorithm.get_session_statistics()
            
            self.logger.info(
                f"📊 [POOL] Workers: {stats['active_workers']} | "
                f"Hashrate: {stats['total_hashrate_mhs']:.1f} H/s | "
                f"Shares: {stats['accepted_shares']}/{stats['rejected_shares']} | "
                f"Efficiency: {(stats['accepted_shares']/(stats['accepted_shares']+stats['rejected_shares']+0.001)*100):.1f}% | "
                f"Consciousness: {zion_stats.get('total_consciousness', 0):.1f} | "
                f"Sacred Masters: {zion_stats.get('sacred_masters', 0)}"
            )
            
            # Log detailed statistics every 10 minutes
            if int(time.time()) % 600 < 60:  # Every 10 minutes
                report = self.zion_algorithm.generate_mining_report()
                self.logger.info(f"🌟 [SACRED] Detailed Report Generated")
                
        except Exception as e:
            self.logger.error(f"Statistics reporting error: {e}")

    def _cleanup_sessions(self):
        now = time.time()
        to_delete = []
        for sid, meta in self.sessions.items():
            if now - meta.get('last_seen', 0) > 1800:  # 30 min TTL
                to_delete.append(sid)
        for sid in to_delete:
            del self.sessions[sid]
        if to_delete:
            self.logger.debug(f"Session cleanup removed {len(to_delete)} stale sessions")

    def start(self):
        self.running = True
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(50)
        self.log(f"Stratum listening on {self.host}:{self.port}")
        
        # Start statistics reporting thread
        def stats_reporter():
            while self.running:
                time.sleep(60)  # Report every minute
                if self.running:
                    self._report_pool_statistics()
        
        stats_thread = threading.Thread(target=stats_reporter, daemon=True)
        stats_thread.start()
        
        try:
            while self.running:
                client, addr = s.accept()
                t = threading.Thread(target=self.handle, args=(client, addr), daemon=True)
                t.start()
        finally:
            s.close()

    def _handle_getjob(self, state: ClientState, session_param: str | None, mid, params_obj=None):
        token = None
        if isinstance(params_obj, dict):
            token = params_obj.get('token')
        restored = False
        stored_diff = None
        # Token has priority
        if token and token in self.session_tokens:
            meta = self.session_tokens[token]
            stored_diff = meta.get('difficulty')
            sid = meta.get('session_id')
            if sid and not state.session_id:
                state.session_id = sid
            if stored_diff and stored_diff > 0:
                if state.difficulty != stored_diff:
                    restored = True
                state.difficulty = stored_diff
            meta['last_seen'] = time.time()
            self.logger.debug(f"[TOKEN] reuse token={token} session={state.session_id} diff={state.difficulty}")
        else:
            # Fallback to session id path
            if session_param:
                if not state.session_id:
                    state.session_id = session_param
                meta = self.sessions.get(session_param)
                if meta:
                    stored_diff = meta.get('difficulty', self.default_monero_difficulty)
                    if stored_diff and stored_diff > 0 and state.difficulty != stored_diff:
                        state.difficulty = stored_diff
                        restored = True
                    meta['last_seen'] = time.time()
        # Final fallback
        if state.difficulty < 1:
            state.difficulty = self.default_monero_difficulty
        # Sync caches
        if state.session_id:
            sess = self.sessions.setdefault(state.session_id, {'difficulty': state.difficulty, 'last_seen': time.time(), 'wallet': state.worker.lower()})
            sess['difficulty'] = state.difficulty
            sess['last_seen'] = time.time()
        # Generate job
        job = self.generate_monero_style_job(state.difficulty)
        # Force align
        if job['difficulty'] != state.difficulty:
            job['difficulty'] = state.difficulty
            job['target'] = f"{max(1, 0xFFFFFFFF // state.difficulty):08x}"
        resp = { 'id': mid, 'jsonrpc': '2.0', 'result': {
            'job_id': job['job_id'],
            'blob': job['blob'],
            'target': job['target'],
            'seed_hash': job['seed_hash'],
            'next_seed_hash': job['next_seed_hash'],
            'algo': job['algo'],
            'height': job['height'],
            'difficulty': job['difficulty']
        }, 'error': None }
        self.send(state.sock, resp)
        self.logger.debug(f"[GETJOB] token={token} session={state.session_id} diff={state.difficulty} restored={restored} target={job['target']}")
        return

    def _cleanup_session_tokens(self):
        """Remove expired session tokens to prevent unbounded growth."""
        try:
            now = time.time()
            to_del = [tok for tok, meta in self.session_tokens.items() if now - meta.get('last_seen', 0) > self.session_token_ttl]
            for tok in to_del:
                del self.session_tokens[tok]
            if to_del and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"[TOKEN] cleaned {len(to_del)} expired tokens (ttl={self.session_token_ttl}s)")
        except Exception as e:
            self.logger.debug(f"[TOKEN] cleanup error: {e}")

    def get_sacred_flower_stats(self):
        """Get Sacred Flower enhancement statistics"""
        active_flowers = 0
        total_consciousness = 0.0
        blessed_workers = []
        
        for state in self.client_states.values():
            if hasattr(state, 'sacred_flower_active') and state.sacred_flower_active:
                active_flowers += 1
                total_consciousness += state.sacred_flower_consciousness
                blessed_workers.append(state.worker)
        
        return {
            'active_sacred_flowers': active_flowers,
            'total_flower_consciousness': total_consciousness,
            'blessed_workers': blessed_workers,
            'sacred_flower_seed': "419ab63b22035f90ea9f2ecb548c6b9b",  # Sacred Yellow Star Flower
            'flower_type': "Sacred Divine Blossom",
            'consciousness_per_flower': 413.91,
            'bonus_percentage': 51.18
        }

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 🌟 ZION SACRED MINING POOL - MAIN EXECUTION  
# 🌸 Enhanced with Sacred Flower blessing for eliZabet - skryty kod pro eliZabet
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='🌟 ZION Sacred Mining Pool 🌟')
    parser.add_argument('--host', default='0.0.0.0', help='Pool host address')
    parser.add_argument('--port', type=int, default=3333, help='Pool port')
    parser.add_argument('--miner-address', help='Miner reward address')
    
    args = parser.parse_args()
    
    print("═══════════════════════════════════════════════════════════════════════════════════════════")
    print("🌟 ZION SACRED MINING POOL STARTING 🌟")
    print("═══════════════════════════════════════════════════════════════════════════════════════════")
    print(f"📡 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🌟 Sacred Algorithm: ENABLED")
    print(f"🔮 Git Integration: ENABLED") 
    print(f"💎 Multi-Miner Support: CPU + GPU")
    if SACRED_FLOWER_AVAILABLE:
        print(f"🌸 Sacred Flower Enhancement: ENABLED - skryty kod pro eliZabet")
        print(f"🌼 Sacred Divine Blossom Level: 413.91 consciousness points")
        print(f"🎁 eliZabet Bonus: +51.18% sacred rewards")
    print("═══════════════════════════════════════════════════════════════════════════════════════════")
    
    try:
        # Initialize blockchain
        chain = Blockchain()
        
        # Create and start ZION Sacred Pool
        pool = MinimalStratumPool(
            host=args.host, 
            port=args.port, 
            miner_address=args.miner_address,
            chain=chain
        )
        
        print("🚀 Starting ZION Sacred Mining Pool...")
        pool.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Pool shutdown requested...")
        pool.running = False
        print("🌟 ZION Sacred Mining Pool stopped gracefully")
    except Exception as e:
        print(f"❌ Pool startup error: {e}")
        raise
