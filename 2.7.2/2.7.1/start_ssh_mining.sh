#!/bin/bash
# 🚀 ZION 2.7.1 SSH Mining Startup Script
# Automatické spuštění těžby na SSH serveru 91.98.122.165
# Datum: 3. října 2025

set -e  # Exit on any error

# Konfigurace
SSH_SERVER="91.98.122.165"
SSH_USER="root"
ZION_DIR="/root/zion-2.7.1"
MINING_ADDRESS="ZxChain...your_mining_address"  # ZMĚŇTE NA VAŠI ADRESU!

# Barvy pro output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 ZION 2.7.1 SSH Mining Startup${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""
echo -e "${YELLOW}📡 Server: ${SSH_SERVER}${NC}"
echo -e "${YELLOW}📁 Directory: ${ZION_DIR}${NC}"
echo -e "${YELLOW}💰 Mining Address: ${MINING_ADDRESS}${NC}"
echo ""

# Funkce pro SSH příkazy
run_ssh_command() {
    local command="$1"
    local description="$2"
    
    echo -e "${BLUE}🔄 ${description}...${NC}"
    ssh "${SSH_USER}@${SSH_SERVER}" "$command"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${description} - DOKONČENO${NC}"
    else
        echo -e "${RED}❌ ${description} - CHYBA${NC}"
        return 1
    fi
}

# Kontrola připojení SSH
echo -e "${BLUE}🔍 Kontrola SSH připojení...${NC}"
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${SSH_USER}@${SSH_SERVER}" echo "SSH OK" 2>/dev/null; then
    echo -e "${GREEN}✅ SSH připojení - OK${NC}"
else
    echo -e "${RED}❌ SSH připojení selhalo - zkouším s heslem...${NC}"
    echo -e "${YELLOW}💡 Zadejte heslo pro server:${NC}"
    ssh -o StrictHostKeyChecking=no "${SSH_USER}@${SSH_SERVER}" echo "SSH OK s heslem"
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ SSH připojení stále selhává!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ SSH připojení s heslem - OK${NC}"
fi

echo ""
echo -e "${PURPLE}🎯 SPOUŠTĚNÍ TĚŽBY NA SSH SERVERU${NC}"
echo -e "${PURPLE}=================================${NC}"

# 1. Přepnutí do ZION adresáře
run_ssh_command "cd ${ZION_DIR} && pwd" "Přechod do ZION adresáře"

# 2. Kontrola Python prostředí
run_ssh_command "cd ${ZION_DIR} && python3 --version" "Kontrola Python prostředí"

# 3. Kontrola ZION CLI
run_ssh_command "cd ${ZION_DIR} && python3 zion_cli.py --help | head -10" "Kontrola ZION CLI"

# 4. Nastavení optimálního algoritmu (Argon2 pro maximální decentralizaci)
run_ssh_command "cd ${ZION_DIR} && python3 zion_cli.py algorithms switch argon2" "Nastavení Argon2 algoritmu"

# 5. Kontrola dostupných algoritmů
run_ssh_command "cd ${ZION_DIR} && python3 zion_cli.py algorithms list" "Seznam dostupných algoritmů"

# 6. Benchmark systému (rychlý test)
echo -e "${BLUE}🔄 Spouštění rychlého benchmarku...${NC}"
run_ssh_command "cd ${ZION_DIR} && timeout 30 python3 zion_cli.py algorithms benchmark || echo 'Benchmark dokončen'" "Rychlý benchmark"

# 7. Spuštění GPU optimizátoru (pokud je k dispozici)
echo -e "${BLUE}🔄 Spouštění GPU optimizátoru...${NC}"
ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR}/mining && python3 zion_gpu_mining_optimizer.py --auto-detect --optimize > gpu_optimizer.log 2>&1 &" 2>/dev/null || echo "GPU optimizer není k dispozici"

# 8. Spuštění těžby
echo ""
echo -e "${GREEN}🚀 SPOUŠTĚNÍ TĚŽBY!${NC}"
echo -e "${GREEN}==================${NC}"

if [ "$MINING_ADDRESS" = "ZxChain...your_mining_address" ]; then
    echo -e "${RED}⚠️  POZOR: Musíte změnit mining adresu!${NC}"
    echo -e "${YELLOW}💡 Upravte proměnnou MINING_ADDRESS v tomto skriptu${NC}"
    echo ""
    echo -e "${CYAN}🔧 Pro získání mining adresy:${NC}"
    echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR} && python3 zion_cli.py wallet addresses'"
    echo ""
    exit 1
fi

# Spuštění těžby na pozadí
echo -e "${BLUE}🔄 Spouštění těžby s adresou: ${MINING_ADDRESS}${NC}"
ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && nohup python3 zion_cli.py mine ${MINING_ADDRESS} > mining.log 2>&1 &"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ TĚŽBA SPUŠTĚNA ÚSPĚŠNĚ!${NC}"
    echo ""
    echo -e "${CYAN}📊 Sledování těžby:${NC}"
    echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR} && tail -f mining.log'"
    echo ""
    echo -e "${CYAN}📈 Status těžby:${NC}"
    echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR} && python3 zion_cli.py stats'"
    echo ""
    echo -e "${CYAN}🔄 Zastavení těžby:${NC}"
    echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'pkill -f \"python3 zion_cli.py mine\"'"
    echo ""
else
    echo -e "${RED}❌ Chyba při spouštění těžby!${NC}"
    exit 1
fi

# Zobrazení aktuálního stavu
echo -e "${PURPLE}📊 AKTUÁLNÍ STAV SERVERU${NC}"
echo -e "${PURPLE}========================${NC}"

# CPU informace
run_ssh_command "nproc && grep 'model name' /proc/cpuinfo | head -1" "CPU informace"

# RAM informace
run_ssh_command "free -h | head -2" "RAM informace"

# Běžící procesy
run_ssh_command "ps aux | grep -E 'python3|zion|mining' | grep -v grep" "Běžící ZION procesy"

echo ""
echo -e "${GREEN}🎉 SSH MINING SETUP DOKONČEN!${NC}"
echo -e "${GREEN}==============================${NC}"
echo ""
echo -e "${CYAN}📱 Užitečné příkazy pro SSH server:${NC}"
echo -e "${YELLOW}# Připojení k serveru:${NC}"
echo -e "ssh ${SSH_USER}@${SSH_SERVER}"
echo ""
echo -e "${YELLOW}# Sledování těžby:${NC}"
echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR} && tail -f mining.log'"
echo ""
echo -e "${YELLOW}# Status blockchain:${NC}"
echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR} && python3 zion_cli.py stats'"
echo ""
echo -e "${YELLOW}# GPU monitoring (pokud je k dispozici):${NC}"
echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR}/mining && tail -f gpu_optimizer.log'"
echo ""
echo -e "${YELLOW}# Zastavení všech procesů:${NC}"
echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'pkill -f zion'"
echo ""
echo -e "${GREEN}🚀 Těžba běží na SSH serveru! Hodně štěstí! 🌟${NC}"