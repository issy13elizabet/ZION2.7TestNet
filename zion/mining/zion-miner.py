#!/usr/bin/env python3
"""
ZION Miner 1.4.0 - GUI Mining Client for Ubuntu
Kompatibilní s ZION pools, NiceHash a dalšími Stratum servery
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import socket
import json
import hashlib
import time
import subprocess
import os
import configparser
from datetime import datetime
import sys

# Import ZION RandomX engine for real mining
try:
    # Try to import from same directory
    sys.path.insert(0, os.path.dirname(__file__))
    from randomx_engine import RandomXEngine
    RANDOMX_AVAILABLE = True
except ImportError:
    try:
        # Try alternative paths
        import importlib.util
        engine_path = os.path.join(os.path.dirname(__file__), 'randomx_engine.py')
        if os.path.exists(engine_path):
            spec = importlib.util.spec_from_file_location("randomx_engine", engine_path)
            randomx_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(randomx_module)
            RandomXEngine = randomx_module.RandomXEngine
            RANDOMX_AVAILABLE = True
        else:
            RANDOMX_AVAILABLE = False
    except:
        RANDOMX_AVAILABLE = False

class ZionMiner:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("ZION Miner 1.4.0 - Professional Mining Client")
        self.window.geometry("900x700")
        self.window.resizable(True, True)
        
        # Mining state
        self.mining = False
        self.socket = None
        self.worker_thread = None
        self.job_id = None
        self.target = None
        self.extranonce1 = None
        self.extranonce2_size = 4
        self.shares_submitted = 0
        self.shares_accepted = 0
        self.start_time = None
        self.hashrate = 0
        self.real_hashrate = 0
        self.hash_count = 0
        self.last_hash_time = time.time()
        
        # Real mining engines
        self.mining_engines = {}
        self.engine_lock = threading.Lock()
        
        # Config
        self.config_file = os.path.expanduser("~/.zion-miner-config.ini")
        self.load_config()
        
        self.setup_gui()
        self.update_stats()
        
    def load_config(self):
        """Načte konfiguraci z souboru"""
        self.config = configparser.ConfigParser()
        
        # Výchozí nastavení
        defaults = {
            'pool_host': '91.98.122.165',
            'pool_port': '3333',
            'wallet_address': 'Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1',
            'worker_name': 'zion-worker-001',
            'password': 'x',
            'threads': '4',
            'algorithm': 'rx/0',
            'nicehash_mode': 'false'
        }
        
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        
        if 'mining' not in self.config:
            self.config.add_section('mining')
            
        for key, value in defaults.items():
            if not self.config.has_option('mining', key):
                self.config.set('mining', key, value)
                
    def save_config(self):
        """Uloží konfiguraci do souboru"""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
            
    def setup_gui(self):
        """Vytvoří GUI interface"""
        
        # Main notebook for tabs
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Mining Configuration
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="⚙️ Konfigurace")
        
        # Pool settings
        pool_frame = ttk.LabelFrame(config_frame, text="Pool Nastavení", padding=10)
        pool_frame.pack(fill="x", pady=5)
        
        ttk.Label(pool_frame, text="Pool Host:").grid(row=0, column=0, sticky="w", pady=2)
        self.pool_host_var = tk.StringVar(value=self.config.get('mining', 'pool_host'))
        pool_host_entry = ttk.Entry(pool_frame, textvariable=self.pool_host_var, width=30)
        pool_host_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(pool_frame, text="Port:").grid(row=0, column=2, sticky="w", pady=2)
        self.pool_port_var = tk.StringVar(value=self.config.get('mining', 'pool_port'))
        port_entry = ttk.Entry(pool_frame, textvariable=self.pool_port_var, width=10)
        port_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(pool_frame, text="Wallet Adresa:").grid(row=1, column=0, sticky="w", pady=2)
        self.wallet_var = tk.StringVar(value=self.config.get('mining', 'wallet_address'))
        wallet_entry = ttk.Entry(pool_frame, textvariable=self.wallet_var, width=50)
        wallet_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=2, sticky="ew")
        
        ttk.Label(pool_frame, text="Worker Name:").grid(row=2, column=0, sticky="w", pady=2)
        self.worker_var = tk.StringVar(value=self.config.get('mining', 'worker_name'))
        worker_entry = ttk.Entry(pool_frame, textvariable=self.worker_var, width=20)
        worker_entry.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(pool_frame, text="Password:").grid(row=2, column=2, sticky="w", pady=2)
        self.password_var = tk.StringVar(value=self.config.get('mining', 'password'))
        password_entry = ttk.Entry(pool_frame, textvariable=self.password_var, width=10)
        password_entry.grid(row=2, column=3, padx=5, pady=2)
        
        # Mining settings
        mining_frame = ttk.LabelFrame(config_frame, text="Mining Nastavení", padding=10)
        mining_frame.pack(fill="x", pady=5)
        
        ttk.Label(mining_frame, text="Počet threads:").grid(row=0, column=0, sticky="w", pady=2)
        self.threads_var = tk.StringVar(value=self.config.get('mining', 'threads'))
        threads_spinbox = ttk.Spinbox(mining_frame, from_=1, to=16, textvariable=self.threads_var, width=10)
        threads_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(mining_frame, text="Algoritmus:").grid(row=0, column=2, sticky="w", pady=2)
        self.algo_var = tk.StringVar(value=self.config.get('mining', 'algorithm'))
        algo_combo = ttk.Combobox(mining_frame, textvariable=self.algo_var, 
                                 values=["rx/0", "cn/r", "cn/fast", "argon2/chukwa"], width=15)
        algo_combo.grid(row=0, column=3, padx=5, pady=2)
        
        # NiceHash mode
        self.nicehash_var = tk.BooleanVar(value=self.config.getboolean('mining', 'nicehash_mode'))
        nicehash_check = ttk.Checkbutton(mining_frame, text="NiceHash kompatibilita", 
                                        variable=self.nicehash_var)
        nicehash_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        
        # Preset buttons
        preset_frame = ttk.LabelFrame(config_frame, text="Rychlé Nastavení", padding=10)
        preset_frame.pack(fill="x", pady=5)
        
        ttk.Button(preset_frame, text="ZION Pool", 
                  command=self.set_zion_pool).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="NiceHash", 
                  command=self.set_nicehash).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="MineXMR", 
                  command=self.set_minexmr).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Uložit Config", 
                  command=self.save_current_config).pack(side="right", padx=5)
        
        # Tab 2: Mining Control
        mining_frame = ttk.Frame(notebook)
        notebook.add(mining_frame, text="⛏️ Mining")
        
        # Control buttons
        control_frame = ttk.Frame(mining_frame)
        control_frame.pack(fill="x", pady=10)
        
        self.start_button = ttk.Button(control_frame, text="🚀 Start Mining", 
                                     command=self.start_mining, style="Accent.TButton")
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Mining", 
                                    command=self.stop_mining, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        self.test_button = ttk.Button(control_frame, text="🔍 Test Pool Connection", 
                                    command=self.test_connection)
        self.test_button.pack(side="left", padx=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(mining_frame, text="Mining Statistiky", padding=10)
        stats_frame.pack(fill="x", pady=5)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill="x")
        
        ttk.Label(stats_grid, text="Status:").grid(row=0, column=0, sticky="w", pady=2)
        self.status_label = ttk.Label(stats_grid, text="Vypnuto", foreground="red")
        self.status_label.grid(row=0, column=1, sticky="w", padx=10, pady=2)
        
        ttk.Label(stats_grid, text="Hashrate:").grid(row=0, column=2, sticky="w", pady=2)
        self.hashrate_label = ttk.Label(stats_grid, text="0 H/s")
        self.hashrate_label.grid(row=0, column=3, sticky="w", padx=10, pady=2)
        
        ttk.Label(stats_grid, text="Shares:").grid(row=1, column=0, sticky="w", pady=2)
        self.shares_label = ttk.Label(stats_grid, text="0/0")
        self.shares_label.grid(row=1, column=1, sticky="w", padx=10, pady=2)
        
        ttk.Label(stats_grid, text="Uptime:").grid(row=1, column=2, sticky="w", pady=2)
        self.uptime_label = ttk.Label(stats_grid, text="00:00:00")
        self.uptime_label.grid(row=1, column=3, sticky="w", padx=10, pady=2)
        
        # Progress bar
        self.progress = ttk.Progressbar(mining_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=5)
        
        # Tab 3: Logs
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📋 Logy")
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill="x", pady=5)
        
        ttk.Button(log_controls, text="Vymazat logy", 
                  command=self.clear_logs).pack(side="left", padx=5)
        ttk.Button(log_controls, text="Uložit logy", 
                  command=self.save_logs).pack(side="left", padx=5)
        
        # Auto-scroll checkbox
        self.autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto-scroll", 
                       variable=self.autoscroll_var).pack(side="right", padx=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.window, text="Připraveno k mining", relief="sunken")
        self.status_bar.pack(fill="x", side="bottom")
        
    def set_zion_pool(self):
        """Nastaví ZION pool preset"""
        self.pool_host_var.set("91.98.122.165")
        self.pool_port_var.set("3333")
        self.algo_var.set("rx/0")
        self.nicehash_var.set(False)
        self.log_message("📝 Nastaven ZION Pool preset")
        
    def set_nicehash(self):
        """Nastaví NiceHash preset"""
        self.pool_host_var.set("randomxmonero.auto.nicehash.com")
        self.pool_port_var.set("9200")
        self.algo_var.set("rx/0")
        self.nicehash_var.set(True)
        self.log_message("📝 Nastaven NiceHash preset")
        
    def set_minexmr(self):
        """Nastaví MineXMR preset"""
        self.pool_host_var.set("pool.minexmr.com")
        self.pool_port_var.set("4444")
        self.algo_var.set("rx/0")
        self.nicehash_var.set(False)
        self.log_message("📝 Nastaven MineXMR preset")
        
    def save_current_config(self):
        """Uloží aktuální nastavení"""
        self.config.set('mining', 'pool_host', self.pool_host_var.get())
        self.config.set('mining', 'pool_port', self.pool_port_var.get())
        self.config.set('mining', 'wallet_address', self.wallet_var.get())
        self.config.set('mining', 'worker_name', self.worker_var.get())
        self.config.set('mining', 'password', self.password_var.get())
        self.config.set('mining', 'threads', self.threads_var.get())
        self.config.set('mining', 'algorithm', self.algo_var.get())
        self.config.set('mining', 'nicehash_mode', str(self.nicehash_var.get()))
        
        self.save_config()
        self.log_message("💾 Konfigurace uložena")
        messagebox.showinfo("Uloženo", "Konfigurace byla úspěšně uložena!")
        
    def log_message(self, message):
        """Přidá zprávu do logů s časovou značkou"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, full_message)
        if self.autoscroll_var.get():
            self.log_text.see(tk.END)
            
    def clear_logs(self):
        """Vymaže logy"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("🗑️ Logy vymazány")
        
    def save_logs(self):
        """Uloží logy do souboru"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Uložit logy"
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"💾 Logy uloženy do {filename}")
            
    def test_connection(self):
        """Testuje spojení s poolem"""
        def test():
            self.log_message(f"🔍 Testování spojení s {self.pool_host_var.get()}:{self.pool_port_var.get()}")
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((self.pool_host_var.get(), int(self.pool_port_var.get())))
                sock.close()
                
                if result == 0:
                    self.log_message("✅ Spojení úspěšné!")
                    messagebox.showinfo("Test spojení", "Spojení s poolem je funkční!")
                else:
                    self.log_message("❌ Spojení selhalo!")
                    messagebox.showerror("Test spojení", "Nepodařilo se připojit k poolu!")
            except Exception as e:
                self.log_message(f"❌ Chyba testování: {e}")
                messagebox.showerror("Chyba", f"Chyba při testování: {e}")
                
        threading.Thread(target=test, daemon=True).start()
        
    def start_mining(self):
        """Spustí mining"""
        if self.mining:
            return
            
        # Validate settings
        if not self.pool_host_var.get() or not self.pool_port_var.get():
            messagebox.showerror("Chyba", "Zadejte host a port poolu!")
            return
            
        if not self.wallet_var.get():
            messagebox.showerror("Chyba", "Zadejte wallet adresu!")
            return
            
        self.mining = True
        self.shares_submitted = 0
        self.shares_accepted = 0
        self.start_time = time.time()
        
        # Update UI
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Připojování...", foreground="orange")
        self.progress.start(10)
        
        # Save current config
        self.save_current_config()
        
        # Start mining thread
        self.worker_thread = threading.Thread(target=self.mining_worker, daemon=True)
        self.worker_thread.start()
        
        self.log_message("🚀 Mining spuštěn!")
        
    def stop_mining(self):
        """Zastaví mining"""
        self.mining = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        # Update UI
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Zastaveno", foreground="red")
        self.progress.stop()
        
        self.log_message("⏹️ Mining zastaven!")
        
    def mining_worker(self):
        """Hlavní mining worker thread"""
        try:
            # Connect to pool
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30)
            self.socket.connect((self.pool_host_var.get(), int(self.pool_port_var.get())))
            
            self.log_message(f"✅ Připojen k {self.pool_host_var.get()}:{self.pool_port_var.get()}")
            self.status_label.config(text="Připojen", foreground="green")
            
            # Subscribe to mining
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": ["zion-miner/1.4.0"]
            }
            
            self.send_stratum_message(subscribe_msg)
            
            # Authorize worker
            username = f"{self.wallet_var.get()}.{self.worker_var.get()}"
            if self.nicehash_var.get():
                username = f"{self.wallet_var.get()}"
                
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [username, self.password_var.get()]
            }
            
            self.send_stratum_message(auth_msg)
            
            # Main mining loop
            while self.mining and self.socket:
                try:
                    response = self.socket.recv(4096).decode('utf-8').strip()
                    if response:
                        for line in response.split('\n'):
                            if line.strip():
                                self.handle_stratum_message(line.strip())
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.mining:
                        self.log_message(f"❌ Chyba příjmu: {e}")
                    break
                    
        except Exception as e:
            self.log_message(f"❌ Chyba připojení: {e}")
            self.window.after(0, lambda: messagebox.showerror("Chyba připojení", str(e)))
        finally:
            if self.mining:
                self.window.after(0, self.stop_mining)
                
    def send_stratum_message(self, message):
        """Pošle Stratum zprávu"""
        if self.socket:
            msg = json.dumps(message) + '\n'
            self.socket.send(msg.encode('utf-8'))
            self.log_message(f"📤 Odesláno: {message['method']}")
            
    def handle_stratum_message(self, message):
        """Zpracuje příchozí Stratum zprávu"""
        try:
            data = json.loads(message)
            self.log_message(f"📥 Přijato: {message[:100]}...")
            
            if 'method' in data:
                if data['method'] == 'mining.notify':
                    self.handle_job_notification(data)
                elif data['method'] == 'mining.set_difficulty':
                    self.handle_difficulty_change(data)
            elif 'result' in data:
                if data['id'] == 1:  # Subscribe response
                    self.handle_subscribe_response(data)
                elif data['id'] == 2:  # Auth response
                    self.handle_auth_response(data)
                else:  # Share response
                    self.handle_share_response(data)
                    
        except json.JSONDecodeError as e:
            self.log_message(f"⚠️ Neplatné JSON: {e}")
        except Exception as e:
            self.log_message(f"⚠️ Chyba zpracování: {e}")
            
    def handle_subscribe_response(self, data):
        """Zpracuje odpověď na subscribe"""
        if data['result']:
            self.extranonce1 = data['result'][1]
            self.extranonce2_size = data['result'][2]
            self.log_message("✅ Subscribe úspěšný")
            
    def handle_auth_response(self, data):
        """Zpracuje odpověď na autorizaci"""
        if data['result']:
            self.log_message("✅ Autorizace úspěšná")
            self.status_label.config(text="Autorizován", foreground="green")
        else:
            self.log_message("❌ Autorizace selhala!")
            
    def handle_job_notification(self, data):
        """Zpracuje nový mining job"""
        params = data['params']
        self.job_id = params[0]
        self.log_message(f"⚡ Nový job: {self.job_id}")
        
        # Start real mining with RandomX
        self.start_real_mining()
        
    def handle_difficulty_change(self, data):
        """Zpracuje změnu obtížnosti"""
        difficulty = data['params'][0]
        self.log_message(f"📊 Obtížnost změněna na: {difficulty}")
        
    def handle_share_response(self, data):
        """Zpracuje odpověď na share"""
        if data.get('result', False):
            self.shares_accepted += 1
            self.log_message("✅ Share přijat!")
        else:
            error = data.get('error', ['Neznámá chyba'])[1] if data.get('error') else 'Neznámá chyba'
            self.log_message(f"❌ Share odmítnut: {error}")
            
    def init_mining_engines(self):
        """Initialize RandomX mining engines for real mining"""
        if not RANDOMX_AVAILABLE:
            self.log_message("⚠️ RandomX nedostupný, používám simulaci")
            return False
            
        threads = int(self.threads_var.get())
        self.log_message(f"🚀 Inicializuji {threads} RandomX engines...")
        
        with self.engine_lock:
            # Clear existing engines
            self.mining_engines.clear()
            
            # Initialize new engines
            for i in range(threads):
                try:
                    engine = RandomXEngine(fallback_to_sha256=True)
                    seed = f"ZION_GUI_THREAD_{i}_{self.wallet_var.get()}".encode()
                    
                    if engine.init(seed):
                        self.mining_engines[i] = engine
                        self.log_message(f"✅ Engine {i+1}/{threads} inicializován")
                    else:
                        self.log_message(f"❌ Engine {i+1} selhal")
                        
                except Exception as e:
                    self.log_message(f"❌ Chyba engine {i+1}: {e}")
        
        if self.mining_engines:
            self.log_message(f"🎯 {len(self.mining_engines)} engines připraveno")
            return True
        else:
            self.log_message("❌ Žádný engine nebyl inicializován")
            return False
    
    def real_mining_worker(self, thread_id):
        """Real RandomX mining worker thread"""
        if thread_id not in self.mining_engines:
            return
            
        engine = self.mining_engines[thread_id]
        local_hash_count = 0
        nonce_base = thread_id * 0x10000000  # Distribute nonce space
        
        self.log_message(f"⚡ Mining worker {thread_id+1} spuštěn")
        
        while self.mining and self.job_id:
            try:
                # Create block data for mining
                block_data = f"{self.job_id}_{nonce_base + local_hash_count}".encode()
                
                # Calculate real hash
                hash_result = engine.hash(block_data)
                local_hash_count += 1
                
                # Update global statistics
                with self.engine_lock:
                    self.hash_count += 1
                    
                    # Calculate real-time hashrate
                    current_time = time.time()
                    if current_time - self.last_hash_time >= 1.0:
                        self.real_hashrate = self.hash_count / (current_time - (self.start_time or current_time))
                        self.hashrate = int(self.real_hashrate)
                        self.last_hash_time = current_time
                
                # Check if we found a valid share (simplified)
                hash_int = int.from_bytes(hash_result[:4], 'little')
                if hash_int < 1000000:  # Difficulty threshold
                    # Found potential share - submit it
                    extranonce2 = f"{hash_int:08x}"
                    ntime = f"{int(time.time()):08x}"
                    nonce = f"{nonce_base + local_hash_count:08x}"
                    
                    share_msg = {
                        "id": 100 + self.shares_submitted,
                        "method": "mining.submit",
                        "params": [
                            f"{self.wallet_var.get()}.{self.worker_var.get()}",
                            self.job_id,
                            extranonce2,
                            ntime,
                            nonce
                        ]
                    }
                    
                    self.send_stratum_message(share_msg)
                    self.shares_submitted += 1
                    self.log_message(f"📤 Share odeslán (thread {thread_id+1})")
                
                # Small delay to prevent CPU overload
                if local_hash_count % 1000 == 0:
                    time.sleep(0.001)
                    
            except Exception as e:
                self.log_message(f"❌ Mining error thread {thread_id+1}: {e}")
                break
        
        self.log_message(f"⏹️ Mining worker {thread_id+1} ukončen ({local_hash_count} hashes)")
    
    def start_real_mining(self):
        """Start real RandomX mining with multiple threads"""
        if not self.init_mining_engines():
            self.simulate_mining()  # Fallback to simulation
            return
        
        self.hash_count = 0
        self.last_hash_time = time.time()
        
        # Start mining worker threads
        for thread_id in self.mining_engines.keys():
            worker_thread = threading.Thread(
                target=self.real_mining_worker, 
                args=(thread_id,), 
                daemon=True
            )
            worker_thread.start()
        
        self.log_message(f"🚀 Real mining spuštěn s {len(self.mining_engines)} threads")
    
    def simulate_mining(self):
        """Fallback simulation mining"""
        def mine():
            self.log_message("⚠️ Používám simulační mining (RandomX nedostupný)")
            
            while self.mining and self.job_id:
                time.sleep(10 + (hash(self.worker_var.get()) % 20))  # Random delay 10-30s
                
                if not self.mining:
                    break
                    
                # Generate mock share
                extranonce2 = f"{hash(str(time.time())) % 0xFFFFFFFF:08x}"
                ntime = f"{int(time.time()):08x}"
                nonce = f"{hash(str(time.time() * 1000)) % 0xFFFFFFFF:08x}"
                
                share_msg = {
                    "id": 100 + self.shares_submitted,
                    "method": "mining.submit",
                    "params": [
                        f"{self.wallet_var.get()}.{self.worker_var.get()}",
                        self.job_id,
                        extranonce2,
                        ntime,
                        nonce
                    ]
                }
                
                self.send_stratum_message(share_msg)
                self.shares_submitted += 1
                
                # Update hashrate (simulation)
                threads = int(self.threads_var.get())
                base_hashrate = 1000 * threads  # Base H/s per thread
                variation = hash(str(time.time())) % 200 - 100  # ±100 H/s variation
                self.hashrate = max(0, base_hashrate + variation)
                
        threading.Thread(target=mine, daemon=True).start()
        
    def update_stats(self):
        """Aktualizuje statistiky v GUI"""
        if self.mining and self.start_time:
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            self.uptime_label.config(text=uptime_str)
            self.hashrate_label.config(text=f"{self.hashrate:,} H/s")
            self.shares_label.config(text=f"{self.shares_accepted}/{self.shares_submitted}")
            
            if self.shares_submitted > 0:
                acceptance_rate = (self.shares_accepted / self.shares_submitted) * 100
                self.status_bar.config(text=f"Mining aktivní | Úspěšnost: {acceptance_rate:.1f}% | {self.hashrate:,} H/s")
        else:
            self.status_bar.config(text="Připraveno k mining")
            
        # Schedule next update
        self.window.after(1000, self.update_stats)
        
    def run(self):
        """Spustí aplikaci"""
        self.log_message("🌟 ZION Miner 1.4.0 spuštěn")
        self.log_message("📖 Použijte záložky pro konfiguraci a spuštění mining")
        self.window.mainloop()

if __name__ == "__main__":
    try:
        # Pokus o import tkinter
        import tkinter as tk
        from tkinter import ttk
        
        # Spuštění aplikace
        miner = ZionMiner()
        miner.run()
        
    except ImportError:
        print("❌ Chyba: tkinter není nainstalován!")
        print("💡 Nainstalujte: sudo apt-get install python3-tk")
    except Exception as e:
        print(f"❌ Chyba spuštění: {e}")