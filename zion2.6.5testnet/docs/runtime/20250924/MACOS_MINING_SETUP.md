# macOS Mining Setup - 2025-09-24

## Dual Platform Mining Configuration

### Windows Status (Current):
- ✅ XMRig běží s 12 vlákny na AMD Ryzen 5 3600
- ✅ SSH tunel na localhost:3333
- ✅ Backend: stabilizace probíhá

### macOS Setup (Nový):

**Hardware předpoklady:**
- Apple Silicon (M1/M2) nebo Intel Mac
- Doporučeno: 8+ GB RAM pro RandomX
- Network: přípojení na 91.98.122.165:3333

**XMRig binární:**
- Cesta: `Zion/mining/platforms/macos-x64/xmrig-6.21.3/xmrig`
- Verze: 6.21.3 (macOS x64 build)
- Architektura: x86_64 (Intel kompatibilita)

**Plánovaná konfigurace:**
```bash
./xmrig \
  --threads auto \
  --url localhost:3333 \
  --coin monero \
  --user MAITREYA_MAC \
  --pass x \
  --donate-level 1 \
  --log-file macos-mining.log
```

**SSH Tunel (sdílený s Windows):**
- Port forward: localhost:3333 → 91.98.122.165:3333
- Existující tunel by měl fungovat pro oba systémy

**Očekávané výsledky:**
- Dual mining: Windows + macOS současně
- Kombinovaný hashrate: Windows_hashrate + macOS_hashrate
- Separate user identifiers: MAITREYA vs MAITREYA_MAC

**Monitoring příkazy (macOS):**
```bash
# Process status
ps aux | grep xmrig

# Network connectivity
nc -zv localhost 3333

# Log monitoring
tail -f macos-mining.log

# Kill mining (if needed)
pkill xmrig
```

**Spuštění sekvence:**
1. Ověřit SSH tunel (sdílený s Windows)
2. Test konektivity na localhost:3333
3. Spustit macOS XMRig s auto-threads
4. Monitor performance a stabilitu
5. Log výsledky do Git

---
**Poznámka:** Tento setup umožní mining z obou platforem současně pro maximální využití dostupných zdrojů.