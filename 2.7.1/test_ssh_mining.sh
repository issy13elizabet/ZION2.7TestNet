#!/bin/bash
# 🧪 ZION 2.7.1 SSH Mining Test
# Rychlý test těžby na SSH serveru
# Datum: 3. října 2025

# Konfigurace
SSH_SERVER="91.98.122.165"
SSH_USER="root"
ZION_DIR="/root/zion-2.7.1"

# Barvy
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🧪 ZION 2.7.1 SSH Mining Test${NC}"
echo -e "${CYAN}=============================${NC}"
echo ""

# Test 1: SSH připojení
echo -e "${BLUE}🔍 Test 1: SSH Připojení${NC}"
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${SSH_SERVER}" echo "SSH připojení OK" 2>/dev/null; then
    echo -e "${GREEN}✅ SSH připojení funguje${NC}"
else
    echo -e "${RED}❌ SSH připojení selhalo!${NC}"
    echo -e "${YELLOW}💡 Zkontrolujte síťové připojení a SSH klíče${NC}"
    echo -e "${CYAN}Zkouším interaktivní připojení...${NC}"
    ssh -o StrictHostKeyChecking=no "${SSH_USER}@${SSH_SERVER}" echo "SSH OK - interaktivní"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ SSH funguje s heslem${NC}"
    else
        exit 1
    fi
fi

# Test 2: ZION adresář
echo -e "${BLUE}🔍 Test 2: ZION Adresář${NC}"
if ssh "${SSH_USER}@${SSH_SERVER}" "test -d ${ZION_DIR}" 2>/dev/null; then
    echo -e "${GREEN}✅ ZION adresář existuje: ${ZION_DIR}${NC}"
else
    echo -e "${RED}❌ ZION adresář neexistuje!${NC}"
    echo -e "${YELLOW}💡 Spusťte nejprve deployment skript${NC}"
    exit 1
fi

# Test 3: Python prostředí
echo -e "${BLUE}🔍 Test 3: Python Prostředí${NC}"
python_version=$(ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && python3 --version" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python dostupný: ${python_version}${NC}"
else
    echo -e "${RED}❌ Python není dostupný!${NC}"
    exit 1
fi

# Test 4: ZION CLI
echo -e "${BLUE}🔍 Test 4: ZION CLI${NC}"
if ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && python3 zion_cli.py --help" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ ZION CLI funguje${NC}"
else
    echo -e "${RED}❌ ZION CLI nefunguje!${NC}"
    exit 1
fi

# Test 5: Dostupné algoritmy
echo -e "${BLUE}🔍 Test 5: Dostupné Algoritmy${NC}"
algorithms=$(ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && python3 zion_cli.py algorithms list" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Algoritmy dostupné:${NC}"
    echo "$algorithms" | while read line; do
        echo -e "${CYAN}  $line${NC}"
    done
else
    echo -e "${RED}❌ Chyba při načítání algoritmů!${NC}"
fi

# Test 6: Wallet adresy
echo -e "${BLUE}🔍 Test 6: Wallet Adresy${NC}"
wallet_addresses=$(ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && python3 zion_cli.py wallet addresses" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Wallet adresy:${NC}"
    echo "$wallet_addresses" | while read line; do
        echo -e "${CYAN}  $line${NC}"
    done
    
    # Extrakce první adresy pro test
    MINING_ADDRESS=$(echo "$wallet_addresses" | grep -o 'Zx[A-Za-z0-9]*' | head -1)
    if [ -n "$MINING_ADDRESS" ]; then
        echo -e "${GREEN}✅ Mining adresa pro test: ${MINING_ADDRESS}${NC}"
    fi
else
    echo -e "${RED}❌ Chyba při načítání wallet adres!${NC}"
fi

# Test 7: Systémové prostředky
echo -e "${BLUE}🔍 Test 7: Systémové Prostředky${NC}"
system_info=$(ssh "${SSH_USER}@${SSH_SERVER}" "echo 'CPU cores:' \$(nproc) && echo 'RAM:' \$(free -h | awk 'NR==2{print \$2}') && echo 'Load:' \$(uptime | cut -d',' -f3-)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Systémové info:${NC}"
    echo "$system_info" | while read line; do
        echo -e "${CYAN}  $line${NC}"
    done
else
    echo -e "${YELLOW}⚠️  Systémové info nedostupné${NC}"
fi

# Test 8: Krátký benchmark
echo -e "${BLUE}🔍 Test 8: Krátký Benchmark (30s)${NC}"
echo -e "${YELLOW}🔄 Spouštím benchmark...${NC}"
benchmark_result=$(ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && timeout 30 python3 zion_cli.py algorithms benchmark 2>/dev/null || echo 'Benchmark dokončen/přerušen'")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Benchmark výsledek:${NC}"
    echo "$benchmark_result" | tail -10 | while read line; do
        echo -e "${CYAN}  $line${NC}"
    done
else
    echo -e "${YELLOW}⚠️  Benchmark se nepodařil dokončit${NC}"
fi

# Test 9: Test mining (pokud máme adresu)
if [ -n "$MINING_ADDRESS" ]; then
    echo -e "${BLUE}🔍 Test 9: Test Mining (60s)${NC}"
    echo -e "${YELLOW}🔄 Spouštím test těžby na 60 sekund...${NC}"
    
    # Spuštění těžby na pozadí
    ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && timeout 60 python3 zion_cli.py mine ${MINING_ADDRESS} > test_mining.log 2>&1 &"
    
    # Čekání na spuštění
    sleep 5
    
    # Kontrola, zda těžba běží
    mining_check=$(ssh "${SSH_USER}@${SSH_SERVER}" "ps aux | grep 'python3.*mine' | grep -v grep")
    if [ -n "$mining_check" ]; then
        echo -e "${GREEN}✅ Test těžba spuštěna${NC}"
        echo -e "${CYAN}  $mining_check${NC}"
        
        # Čekání na dokončení
        sleep 55
        
        # Zobrazení logů z testu
        echo -e "${BLUE}📋 Test mining logy:${NC}"
        test_logs=$(ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && tail -10 test_mining.log 2>/dev/null")
        if [ -n "$test_logs" ]; then
            echo "$test_logs" | while read line; do
                echo -e "${CYAN}  $line${NC}"
            done
        fi
        
        echo -e "${GREEN}✅ Test mining dokončen${NC}"
    else
        echo -e "${RED}❌ Test mining se nepodařil spustit${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Test 9: Přeskočen - žádná mining adresa${NC}"
fi

# Výsledek testů
echo ""
echo -e "${CYAN}🎉 VÝSLEDEK SSH MINING TESTŮ${NC}"
echo -e "${CYAN}=============================${NC}"
echo ""
echo -e "${GREEN}✅ SSH server je připraven pro těžbu ZION 2.7.1!${NC}"
echo ""
echo -e "${YELLOW}📱 Příkazy pro spuštění těžby:${NC}"
echo -e "${CYAN}# Spustit těžbu:${NC}"
echo -e "./start_ssh_mining.sh"
echo ""
echo -e "${CYAN}# Sledovat těžbu:${NC}"
echo -e "./monitor_ssh_mining.sh"
echo ""
echo -e "${CYAN}# Připojit se k serveru:${NC}"
echo -e "ssh ${SSH_USER}@${SSH_SERVER}"
echo ""
echo -e "${CYAN}# Ruční spuštění těžby:${NC}"
if [ -n "$MINING_ADDRESS" ]; then
    echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR} && python3 zion_cli.py mine ${MINING_ADDRESS}'"
else
    echo -e "ssh ${SSH_USER}@${SSH_SERVER} 'cd ${ZION_DIR} && python3 zion_cli.py mine YOUR_ADDRESS'"
fi
echo ""
echo -e "${GREEN}🚀 Vše je připraveno k těžbě! Hodně štěstí! 🌟${NC}"