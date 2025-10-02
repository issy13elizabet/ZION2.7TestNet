"""
ZION 2.7 Stratum Pool (Integrated With Blockchain)
Real socket, real block templates, basic share -> block submission.
"""
import socket, threading, json, time, hashlib, os, sys
from dataclasses import dataclass, field
from datetime import datetime

# Import blockchain core
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from core.blockchain import Blockchain, Consensus

@dataclass
class ClientState:
    sock: socket.socket
    address: tuple
    authorized: bool = False
    worker: str = "unknown"
    difficulty: int = 1
    last_adjust: float = field(default_factory=time.time)
    shares_since_adjust: int = 0
    accepted_shares: int = 0
    rejected_shares: int = 0
    blocks_found: int = 0

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
        # Very simple varDiff parameters
        self.vardiff_enabled = True
        self.vardiff_target_share_time = 15.0  # seconds/share target
        self.vardiff_adjust_window = 60.0      # seconds between adjustments
        self.vardiff_max_jump = 2              # cap doubling/halving rate

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
        new_diff = state.difficulty
        if avg_share_time < target * 0.5:
            new_diff = min(state.difficulty * 2, state.difficulty * self.vardiff_max_jump)
        elif avg_share_time > target * 1.8:
            new_diff = max(1, state.difficulty // 2)
        if new_diff != state.difficulty:
            self._set_client_difficulty(state, new_diff)
        state.last_adjust = now
        state.shares_since_adjust = 0

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
                    self._process(client, addr, line)
            except Exception:
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
        if method == 'mining.subscribe':
            resp = { 'id': mid, 'result': [[['mining.set_difficulty','1'],['mining.notify','1']], '0001', 4], 'error': None }
            self.send(client, resp)
            job = self.generate_job()
            notify = { 'id': None, 'method': 'mining.notify', 'params': [ job['job_id'], job['prev_hash'], job['coinbase1'], job['coinbase2'], job['merkle_branch'], job['version'], job['nbits'], job['ntime'], job['clean_jobs'] ] }
            self.send(client, notify)
            # initialize difficulty (can diverge via varDiff later)
            self._set_client_difficulty(state, max(1, self.chain.current_difficulty))
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
                        self.accepted += 1
                        self.send(client, { 'id': mid, 'result': True, 'error': None })
                        self.log(f"BLOCK ACCEPTED height={mined_block['height']} new_height={self.chain.height()} hash={self.chain.last_block().hash[:16]}... worker={state.worker}")
                        new_job = self.generate_job()
                        notify = { 'id': None, 'method': 'mining.notify', 'params': [ new_job['job_id'], new_job['prev_hash'], new_job['coinbase1'], new_job['coinbase2'], new_job['merkle_branch'], new_job['version'], new_job['nbits'], new_job['ntime'], new_job['clean_jobs'] ] }
                        # broadcast notify to all connected clients
                        for cs in list(self.client_states.values()):
                            try:
                                self.send(cs.sock, notify)
                            except Exception:
                                pass
                    else:
                        self.rejected += 1
                        self.send(client, { 'id': mid, 'result': False, 'error': 'Block rejected' })
                else:
                    self.send(client, { 'id': mid, 'result': True, 'error': None })
                self._maybe_adjust_vardiff(state)
            else:
                state.rejected_shares += 1
                self.rejected_shares += 1
                self.rejected += 1
                self.send(client, { 'id': mid, 'result': False, 'error': 'Low difficulty share' })
        else:
            self.send(client, { 'id': mid, 'result': None, 'error': f'Unknown method {method}' })

    def start(self):
        self.running = True
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(50)
        self.log(f"Stratum listening on {self.host}:{self.port}")
        try:
            while self.running:
                client, addr = s.accept()
                t = threading.Thread(target=self.handle, args=(client, addr), daemon=True)
                t.start()
        finally:
            s.close()

if __name__ == '__main__':
    pool = MinimalStratumPool()
    pool.start()
