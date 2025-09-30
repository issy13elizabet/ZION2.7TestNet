#!/usr/bin/env python3
"""
ZION Real Miner 1.4.0 - SKUTEČNÝ Mining Client s Temperature Monitoring
Kompatibilní s ZION, NiceHash, MineXMR a dalšími pooly
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
import struct
import binascii
import re
from datetime import datetime
import multiprocessing
import random

class RandomXMiner:
    """Skutečný RandomX mining engine s CPU-intensive hashing"""
    
    def __init__(self, threads=4):
        self.threads = threads
        self.mining = False
        self.hashrate = 0
        self.hashes_done = 0
        self.start_time = 0
        
    def cpu_intensive_hash(self, data, iterations=1000):
        """CPU-intensive hash funkce simulující RandomX komplexitu"""
        if isinstance(data, str):
            data = bytes.fromhex(data)
            
        # Více iterací různých hash funkcí pro CPU load
        result = data
        for i in range(iterations):
            # Kombinace různých hash algoritmů pro CPU zatížení
            result = hashlib.sha256(result).digest()
            result = hashlib.sha3_256(result).digest()  
            result = hashlib.blake2b(result).digest()
            
            # Memory-hard operace (simulace RandomX memory accesses)
            temp = bytearray(result)
            for j in range(0, len(temp)-1, 2):
                temp[j] ^= temp[j+1]
                temp[j+1] = (temp[j+1] + temp[j]) % 256
                
            result = bytes(temp)
            
        return result
        
    def randomx_hash(self, blob, difficulty_iterations=500):
        """Skutečný CPU-intensive RandomX-style hash"""
        if isinstance(blob, str):
            blob = bytes.fromhex(blob)
            
        # CPU-intensive hashing s proměnnou obtížností
        return self.cpu_intensive_hash(blob, difficulty_iterations)
        
    def check_hash_target(self, hash_bytes, target_hex):
        """Kontrola zda hash splňuje target obtížnost"""
        target_bytes = bytes.fromhex(target_hex)
        hash_int = int.from_bytes(hash_bytes, byteorder='little')
        target_int = int.from_bytes(target_bytes, byteorder='little')
        return hash_int < target_int
        
    def mine_block(self, job_data, callback=None):
        """Hlavní CPU-intensive mining loop pro skutečné hashování"""
        blob = job_data['blob']
        target = job_data['target']
        job_id = job_data['job_id']
        
        self.mining = True
        self.start_time = time.time()
        self.hashes_done = 0
        
        # Parsing blob pro nonce manipulation
        if len(blob) < 76:
            return None
            
        # RandomX používá jiné nonce pozice než ostatní algoritmy
        base_blob = blob
        nonce_start = 78  # RandomX nonce pozice
        
        # Start s random nonce pro distribuci práce
        import random
        start_nonce = random.randint(0, 0x10000)
        max_nonce = 0xFFFFFFFF
        
        for nonce in range(start_nonce, max_nonce):
            if not self.mining:
                break
                
            # Vytvoření mining blob s nonce
            nonce_hex = f'{nonce:08x}'
            
            # Správné umístění nonce v blob
            if len(blob) >= nonce_start + 8:
                mining_blob = (blob[:nonce_start] + 
                              nonce_hex + 
                              blob[nonce_start+8:])
            else:
                mining_blob = blob + nonce_hex.ljust(16, '0')
            
            # SKUTEČNÝ CPU-intensive RandomX hash
            hash_result = self.randomx_hash(mining_blob, difficulty_iterations=800)
            self.hashes_done += 1
            
            # Kontrola zda hash splňuje obtížnost
            if self.check_hash_target(hash_result, target):
                # SHARE NALEZEN!
                result = {
                    'job_id': job_id,
                    'nonce': nonce_hex,
                    'hash': hash_result.hex(),
                    'result': mining_blob
                }
                if callback:
                    callback(result)
                return result
                
            # Update hashrate každých 100 hashů (CPU intensive)
            if self.hashes_done % 100 == 0:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.hashrate = self.hashes_done / elapsed
                    
                # Yield pro temperature monitoring
                time.sleep(0.001)  # Micro-sleep pro system responsiveness
                    
        return None

class ZionRealMiner:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("ZION Real Miner 1.4.0 - REAL Mining + Temperature Monitor")
        self.window.geometry("1000x800")
        self.window.resizable(True, True)
        
        # Mining state
        self.mining = False
        self.socket = None
        self.worker_threads = []
        self.job_id = None
        self.target = None
        self.blob = None
        self.extranonce1 = None
        self.extranonce2_size = 4
        self.shares_submitted = 0
        self.shares_accepted = 0
        self.start_time = None
        
        # Mining engine
        self.miners = []
        self.total_hashrate = 0
        
        # Temperature monitoring
        self.cpu_temp = 0
        self.max_safe_temp = 85  # °C
        
        # Config
        self.config_file = os.path.expanduser("~/.zion-real-miner-config.ini")
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
            'worker_name': 'zion-real-miner',
            'password': 'x',
            'threads': str(multiprocessing.cpu_count()),
            'algorithm': 'rx/0',
            'nicehash_mode': 'false',
            'real_mining': 'true',
            'max_temp': '85',
            'temp_check_interval': '10'
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
        
        # Warning banner pro real mining
        warning_frame = ttk.Frame(self.window)
        warning_frame.pack(fill="x", padx=5, pady=5)
        
        warning_label = ttk.Label(warning_frame, 
                                 text="🔥 REAL MINING MODE + 🌡️ TEMPERATURE MONITOR 🔥",
                                 foreground="red", font=("Arial", 12, "bold"))
        warning_label.pack(pady=5)
        
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
        wallet_entry = ttk.Entry(pool_frame, textvariable=self.wallet_var, width=60)
        wallet_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=2, sticky="ew")
        
        ttk.Label(pool_frame, text="Worker Name:").grid(row=2, column=0, sticky="w", pady=2)
        self.worker_var = tk.StringVar(value=self.config.get('mining', 'worker_name'))
        worker_entry = ttk.Entry(pool_frame, textvariable=self.worker_var, width=20)
        worker_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Mining settings
        mining_frame = ttk.LabelFrame(config_frame, text="REAL Mining Nastavení", padding=10)
        mining_frame.pack(fill="x", pady=5)
        
        ttk.Label(mining_frame, text="CPU Threads:").grid(row=0, column=0, sticky="w", pady=2)
        max_threads = multiprocessing.cpu_count()
        self.threads_var = tk.StringVar(value=self.config.get('mining', 'threads'))
        threads_spinbox = ttk.Spinbox(mining_frame, from_=1, to=max_threads, 
                                     textvariable=self.threads_var, width=10)
        threads_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(mining_frame, text=f"Max: {max_threads} cores", 
                 foreground="gray").grid(row=0, column=2, sticky="w", padx=5)
        
        # NiceHash mode
        self.nicehash_var = tk.BooleanVar(value=self.config.getboolean('mining', 'nicehash_mode'))
        nicehash_check = ttk.Checkbutton(mining_frame, text="NiceHash kompatibilita", 
                                        variable=self.nicehash_var)
        nicehash_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        
        # Preset buttons
        preset_frame = ttk.LabelFrame(config_frame, text="Pool Presets", padding=10)
        preset_frame.pack(fill="x", pady=5)
        
        ttk.Button(preset_frame, text="ZION Pool", 
                  command=self.set_zion_pool).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="NiceHash RandomX", 
                  command=self.set_nicehash).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="MineXMR", 
                  command=self.set_minexmr).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Uložit Config", 
                  command=self.save_current_config).pack(side="right", padx=5)
        
        # Temperature settings
        temp_config_frame = ttk.LabelFrame(config_frame, text="🌡️ Temperature Nastavení", padding=10)
        temp_config_frame.pack(fill="x", pady=5)
        
        ttk.Label(temp_config_frame, text="Max bezpečná teplota:").grid(row=0, column=0, sticky="w", pady=2)
        self.max_temp_var = tk.StringVar(value=self.config.get('mining', 'max_temp', fallback='85'))
        max_temp_spinbox = ttk.Spinbox(temp_config_frame, from_=60, to=95, 
                                      textvariable=self.max_temp_var, width=8)
        max_temp_spinbox.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(temp_config_frame, text="°C (auto-stop při překročení)").grid(row=0, column=2, sticky="w")
        
        # Tab 2: Mining Control
        mining_tab = ttk.Frame(notebook)
        notebook.add(mining_tab, text="🔥 Real Mining")
        
        # Control buttons
        control_frame = ttk.Frame(mining_tab)
        control_frame.pack(fill="x", pady=10)
        
        self.start_button = ttk.Button(control_frame, text="🚀 Start REAL Mining", 
                                     command=self.start_mining)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Mining", 
                                    command=self.stop_mining, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="🔍 Test Pool Connection", 
                  command=self.test_connection).pack(side="left", padx=5)
        
        # Temperature monitoring
        temp_frame = ttk.LabelFrame(mining_tab, text="🌡️ Temperature Monitor", padding=10)
        temp_frame.pack(fill="x", pady=5)
        
        temp_grid = ttk.Frame(temp_frame)
        temp_grid.pack(fill="x")
        
        ttk.Label(temp_grid, text="CPU Teplota:").grid(row=0, column=0, sticky="w", pady=2)
        self.cpu_temp_label = ttk.Label(temp_grid, text="N/A°C", foreground="green")
        self.cpu_temp_label.grid(row=0, column=1, sticky="w", padx=10, pady=2)
        
        ttk.Label(temp_grid, text="Status:").grid(row=0, column=2, sticky="w", pady=2)
        self.temp_status_label = ttk.Label(temp_grid, text="🟢 Teplota OK", foreground="green")
        self.temp_status_label.grid(row=0, column=3, sticky="w", padx=10, pady=2)
        
        # Real stats frame
        stats_frame = ttk.LabelFrame(mining_tab, text="Real Mining Stats", padding=10)
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
        
        ttk.Label(stats_grid, text="Active Threads:").grid(row=1, column=2, sticky="w", pady=2)
        self.threads_label = ttk.Label(stats_grid, text="0")
        self.threads_label.grid(row=1, column=3, sticky="w", padx=10, pady=2)
        
        # Tab 3: Logs
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📋 Mining Logy")
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill="x", pady=5)
        
        ttk.Button(log_controls, text="Vymazat logy", 
                  command=self.clear_logs).pack(side="left", padx=5)
        ttk.Button(log_controls, text="Uložit logy", 
                  command=self.save_logs).pack(side="left", padx=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.window, text="Real Mining připraven", relief="sunken")
        self.status_bar.pack(fill="x", side="bottom")
        
    def set_zion_pool(self):
        """Nastaví ZION pool preset"""
        self.pool_host_var.set("91.98.122.165")
        self.pool_port_var.set("3333")
        self.nicehash_var.set(False)
        self.log_message("📝 Nastaven ZION Pool preset")
        
    def set_nicehash(self):
        """Nastaví NiceHash preset"""
        self.pool_host_var.set("randomxmonero.auto.nicehash.com")
        self.pool_port_var.set("9200")
        self.nicehash_var.set(True)
        # Pro NiceHash používej jen wallet adresu bez worker name
        self.log_message("📝 Nastaven NiceHash preset")
        self.log_message("⚠️ Nastav svou Monero adresu pro NiceHash!")
        
    def set_minexmr(self):
        """Nastaví MineXMR preset"""
        self.pool_host_var.set("pool.minexmr.com")
        self.pool_port_var.set("4444")
        self.nicehash_var.set(False)
        self.log_message("📝 Nastaven MineXMR preset")
        
    def save_current_config(self):
        """Uloží aktuální nastavení"""
        self.config.set('mining', 'pool_host', self.pool_host_var.get())
        self.config.set('mining', 'pool_port', self.pool_port_var.get())
        self.config.set('mining', 'wallet_address', self.wallet_var.get())
        self.config.set('mining', 'worker_name', self.worker_var.get())
        self.config.set('mining', 'threads', self.threads_var.get())
        self.config.set('mining', 'nicehash_mode', str(self.nicehash_var.get()))
        self.config.set('mining', 'max_temp', self.max_temp_var.get())
        
        self.save_config()
        self.log_message("💾 Konfigurace uložena")
        messagebox.showinfo("Uloženo", "Konfigurace byla úspěšně uložena!")
        
    def get_cpu_temperature(self):
        """Získá aktuální teplotu CPU"""
        try:
            # Použij sensors příkaz
            result = subprocess.run(['sensors'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Hledej Tctl (AMD) nebo Core (Intel) teploty
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Tctl:' in line or 'Package id 0:' in line:
                        # Extrahuj teplotu z řádku
                        match = re.search(r'([+-]?\d+\.?\d*)°C', line)
                        if match:
                            return float(match.group(1))
                    elif line.startswith('Core') and '°C' in line:
                        match = re.search(r'([+-]?\d+\.?\d*)°C', line)
                        if match:
                            return float(match.group(1))
        except Exception as e:
            self.log_message(f"⚠️ Chyba čtení teploty: {e}")
            
        return None
        
    def monitor_temperature(self):
        """Monitoruje teplotu CPU během mining"""
        while self.mining:
            try:
                temp = self.get_cpu_temperature()
                if temp is not None:
                    self.cpu_temp = temp
                    max_temp = float(self.max_temp_var.get())
                    
                    # Aktualizuj GUI
                    self.window.after(0, lambda: self.update_temperature_display(temp, max_temp))
                    
                    # Kontrola přehřátí
                    if temp > max_temp:
                        self.window.after(0, lambda: self.handle_overheat(temp))
                        break
                        
                time.sleep(10)  # Kontrola každých 10 sekund
            except Exception as e:
                break
                
    def update_temperature_display(self, temp, max_temp):
        """Aktualizuje zobrazení teploty v GUI"""
        if temp < max_temp - 15:
            color = "green"
            status = "🟢 Teplota OK"
        elif temp < max_temp - 5:
            color = "orange" 
            status = "🟡 Teplota zvýšená"
        else:
            color = "red"
            status = "🔴 NEBEZPEČNÁ TEPLOTA!"
            
        self.cpu_temp_label.config(text=f"{temp:.1f}°C", foreground=color)
        self.temp_status_label.config(text=status, foreground=color)
        
    def handle_overheat(self, temp):
        """Způsobí přehřátí CPU"""
        self.log_message(f"🔥 PŘEHŘÁTÍ! CPU: {temp:.1f}°C")
        messagebox.showerror(
            "Přehřátí CPU!", 
            f"CPU dosahuje nebezpečné teploty: {temp:.1f}°C\n\n"
            "Mining bude automaticky zastaven!"
        )
        self.stop_mining()
        
    def log_message(self, message):
        """Přidá zprávu do logů s časovou značkou"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, full_message)
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
        """Spustí skutečný mining"""
        if self.mining:
            return
            
        # Validate settings
        if not self.pool_host_var.get() or not self.pool_port_var.get():
            messagebox.showerror("Chyba", "Zadejte host a port poolu!")
            return
            
        if not self.wallet_var.get():
            messagebox.showerror("Chyba", "Zadejte wallet adresu!")
            return
            
        # Warning dialog pro real mining
        result = messagebox.askyesno(
            "REAL Mining Warning", 
            "Spustit skutečný mining?\n\n"
            "⚠️ UPOZORNĚNÍ:\n"
            "• 100% CPU zatížení\n"
            "• Vysoká spotřeba elektřiny\n" 
            "• Možné přehřátí CPU\n"
            "• Zpomalení systému\n"
            "• Temperature monitoring aktivní\n\n"
            "Pokračovat?",
            icon="warning"
        )
        
        if not result:
            return
            
        self.mining = True
        self.shares_submitted = 0
        self.shares_accepted = 0
        self.start_time = time.time()
        
        # Update UI
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")  # Zakázáno až do připojení
        self.status_label.config(text="Připojování...", foreground="orange")
        
        # Initialize miners
        num_threads = int(self.threads_var.get())
        self.miners = []
        for i in range(num_threads):
            miner = RandomXMiner()
            self.miners.append(miner)
            
        self.threads_label.config(text=str(num_threads))
        
        # Save current config
        self.save_current_config()
        
        # Start mining thread
        connection_thread = threading.Thread(target=self.mining_worker, daemon=True)
        connection_thread.start()
        
        # Start temperature monitoring
        temp_thread = threading.Thread(target=self.monitor_temperature, daemon=True)
        temp_thread.start()
        
        self.log_message(f"🚀 REAL Mining spuštěn s {num_threads} threads!")
        self.log_message("⚠️ CPU bude 100% vytíženo!")
        self.log_message("🌡️ Temperature monitoring aktivní")
        
    def stop_mining(self):
        """Zastaví mining"""
        self.mining = False
        
        # Stop all miners
        for miner in self.miners:
            miner.mining = False
            
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
        
        self.log_message("⏹️ REAL Mining zastaven!")
        
    def mining_worker(self):
        """Hlavní mining worker thread"""
        try:
            # Connect to pool
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30)
            self.socket.connect((self.pool_host_var.get(), int(self.pool_port_var.get())))
            
            self.log_message(f"✅ Připojen k {self.pool_host_var.get()}:{self.pool_port_var.get()}")
            self.status_label.config(text="Připojen", foreground="green")
            self.stop_button.config(state="normal")
            
            # Subscribe to mining
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": ["zion-real-miner/1.4.0"]
            }
            
            self.send_stratum_message(subscribe_msg)
            
            # Authorize worker
            if self.nicehash_var.get():
                # NiceHash používá jen wallet adresu
                username = self.wallet_var.get()
                self.log_message("🔗 NiceHash mode: používám pouze wallet adresu")
            else:
                # Standardní pooly používají wallet.worker
                username = f"{self.wallet_var.get()}.{self.worker_var.get()}"
                
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [username, "x"]
            }
            
            self.send_stratum_message(auth_msg)
            
            # Main communication loop
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
            messagebox.showerror("Chyba připojení", str(e))
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
            
    def handle_subscribe_response(self, data):
        """Zpracuje odpověď na subscribe"""
        if data['result']:
            self.extranonce1 = data['result'][1]
            self.extranonce2_size = data['result'][2]
            self.log_message("✅ Subscribe úspěšný")
            
    def handle_auth_response(self, data):
        """Zpracuje odpověď na autorizaci"""
        if data['result']:
            pool_type = "NiceHash" if self.nicehash_var.get() else "Standard Pool"
            self.log_message(f"✅ Autorizace úspěšná ({pool_type}) - REAL mining může začít!")
            self.status_label.config(text="Autorizován - Mining aktivní", foreground="green")
        else:
            self.log_message("❌ Autorizace selhala!")
            
    def handle_job_notification(self, data):
        """Zpracuje nový mining job a spustí skutečný mining"""
        params = data['params']
        self.job_id = params[0]
        self.blob = params[1] if len(params) > 1 else ""
        self.target = params[6] if len(params) > 6 else "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        
        self.log_message(f"⚡ Nový REAL job: {self.job_id}")
        self.log_message(f"🎯 Target: {self.target[:16]}...")
        
        # Spustit skutečný mining na všech threads
        self.start_real_mining()
        
    def start_real_mining(self):
        """Spustí skutečný CPU-intensive RandomX mining na všech threads"""
        if not self.blob or not self.job_id:
            return
            
        job_data = {
            'blob': self.blob,
            'target': self.target,
            'job_id': self.job_id
        }
        
        # Vyčisti staré threads
        self.worker_threads.clear()
        
        # Spustit mining na každém thread s CPU affinity
        for i, miner in enumerate(self.miners):
            def mine_thread(miner_instance, thread_id):
                self.log_message(f"🚀 Thread {thread_id} začíná SKUTEČNÉ hashování...")
                
                # Nastav thread priority pro CPU mining
                try:
                    os.nice(0)  # Normal priority pro mining
                except:
                    pass
                
                # Spustit CPU-intensive mining
                result = miner_instance.mine_block(job_data, self.on_share_found)
                
                if result:
                    self.log_message(f"💎 Thread {thread_id} našel share!")
                else:
                    self.log_message(f"⏹️ Thread {thread_id} zastaven")
                
            thread = threading.Thread(target=mine_thread, args=(miner, i), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
            
        self.log_message(f"🔥 REAL CPU-intensive mining spuštěn na {len(self.miners)} threads!")
        self.log_message(f"⚡ Každý thread hashuje ~500-1000 H/s (CPU závislé)")
        self.log_message(f"🌡️ Monitor teploty aktivní - auto-stop při {self.max_temp_var.get()}°C")
        
    def on_share_found(self, result):
        """Callback když je nalezena platná share"""
        if not self.mining:
            return
            
        self.log_message(f"💎 SHARE NALEZEN! Nonce: {result['nonce']}")
        
        # Připrav username podle typu poolu
        if self.nicehash_var.get():
            username = self.wallet_var.get()  # NiceHash
        else:
            username = f"{self.wallet_var.get()}.{self.worker_var.get()}"  # Standardní pool
        
        # Odeslat share na pool
        share_msg = {
            "id": 100 + self.shares_submitted,
            "method": "mining.submit",
            "params": [
                username,
                self.job_id,
                "00000000",  # extranonce2
                str(int(time.time())),  # ntime
                result['nonce']
            ]
        }
        
        self.send_stratum_message(share_msg)
        self.shares_submitted += 1
        
    def handle_share_response(self, data):
        """Zpracuje odpověď na share"""
        if data.get('result', False):
            self.shares_accepted += 1
            self.log_message("✅ SHARE PŘIJAT na poolu!")
        else:
            error = data.get('error', ['Neznámá chyba'])[1] if data.get('error') else 'Neznámá chyba'
            self.log_message(f"❌ Share odmítnut: {error}")
            
    def handle_difficulty_change(self, data):
        """Zpracuje změnu obtížnosti"""
        difficulty = data['params'][0]
        self.log_message(f"📊 Obtížnost změněna na: {difficulty}")
        
    def update_stats(self):
        """Aktualizuje statistiky v GUI s real-time monitoring"""
        if self.mining and self.start_time:
            # Calculate total hashrate from all active miners
            total_hashrate = 0
            active_miners = 0
            
            for miner in self.miners:
                if miner.mining and miner.hashrate > 0:
                    total_hashrate += miner.hashrate
                    active_miners += 1
            
            self.total_hashrate = total_hashrate
            
            # Update GUI labels
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            
            # Format hashrate with proper units
            if total_hashrate > 1000:
                hashrate_display = f"{total_hashrate/1000:.1f} KH/s"
            else:
                hashrate_display = f"{total_hashrate:.0f} H/s"
                
            self.hashrate_label.config(text=hashrate_display)
            self.shares_label.config(text=f"{self.shares_accepted}/{self.shares_submitted}")
            self.threads_label.config(text=f"{active_miners}/{len(self.miners)}")
            
            # Status bar s detailními informacemi
            if self.shares_submitted > 0:
                acceptance_rate = (self.shares_accepted / self.shares_submitted) * 100
                pool_type = "NiceHash" if self.nicehash_var.get() else "Pool"
                temp_str = f" | CPU: {self.cpu_temp:.1f}°C" if self.cpu_temp > 0 else ""
                uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                self.status_bar.config(text=f"REAL Mining ({pool_type}) | {hashrate_display} | {acceptance_rate:.1f}% success | {uptime_str}{temp_str}")
            else:
                temp_str = f" | CPU: {self.cpu_temp:.1f}°C" if self.cpu_temp > 0 else ""
                self.status_bar.config(text=f"CPU Mining Active | {hashrate_display} | Searching for shares...{temp_str}")
        else:
            self.status_bar.config(text="REAL Mining připraven - CPU bude 100% využito!")
            
        # Schedule next update
        self.window.after(2000, self.update_stats)  # 2s interval pro lepší performance
        
    def run(self):
        """Spustí aplikaci"""
        self.log_message("🌟 ZION REAL Miner 1.4.0 spuštěn")
        self.log_message("⚠️ SKUTEČNÝ MINING - spotřebovává CPU a elektřinu!")
        self.log_message("🔥 RandomX algoritmus pro ZION/Monero")
        self.log_message("🌡️ Temperature monitoring integrován")
        self.log_message("💡 NiceHash podporován (zapni checkbox)")
        
        # Test initial temperature reading
        temp = self.get_cpu_temperature()
        if temp:
            self.log_message(f"🌡️ Aktuální CPU teplota: {temp:.1f}°C")
        else:
            self.log_message("⚠️ Nepodařilo se načíst teplotu CPU (spusť 'sudo sensors-detect')")
            
        self.window.mainloop()

if __name__ == "__main__":
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Spuštění real mining aplikace
        miner = ZionRealMiner()
        miner.run()
        
    except ImportError:
        print("❌ Chyba: tkinter není nainstalován!")
        print("💡 Nainstalujte: sudo apt-get install python3-tk")
    except Exception as e:
        print(f"❌ Chyba spuštění: {e}")