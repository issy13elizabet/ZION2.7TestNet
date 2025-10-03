#!/bin/bash
# 📊 ZION 2.7.1 SSH Mining Monitor
# Monitoring těžby a výkonu na SSH serveru
# Datum: 3. října 2025

set -e

# Konfigurace
SSH_SERVER="91.98.122.165"
SSH_USER="root"
ZION_DIR="/root/zion-2.7.1"
UPDATE_INTERVAL=30  # sekund

# Barvy
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Funkce pro vyčištění obrazovky
clear_screen() {
    clear
    echo -e "${CYAN}🚀 ZION 2.7.1 SSH Mining Monitor${NC}"
    echo -e "${CYAN}=================================${NC}"
    echo -e "${YELLOW}📡 Server: ${SSH_SERVER} | 🔄 Update každých ${UPDATE_INTERVAL}s${NC}"
    echo -e "${YELLOW}⏰ $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
}

# Funkce pro SSH příkaz s timeout
ssh_command() {
    local command="$1"
    timeout 10 ssh "${SSH_USER}@${SSH_SERVER}" "$command" 2>/dev/null || echo "N/A"
}

# Funkce pro zobrazení statistik
show_mining_stats() {
    echo -e "${PURPLE}📊 MINING STATISTIKY${NC}"
    echo -e "${PURPLE}===================${NC}"
    
    # ZION blockchain stats
    local stats=$(ssh_command "cd ${ZION_DIR} && python3 zion_cli.py stats 2>/dev/null | head -20")
    if [ "$stats" != "N/A" ]; then
        echo -e "${GREEN}✅ ZION Stats:${NC}"
        echo "$stats" | while read line; do
            echo -e "${CYAN}  $line${NC}"
        done
    else
        echo -e "${RED}❌ ZION stats nedostupné${NC}"
    fi
    
    echo ""
    
    # Mining procesy
    local mining_processes=$(ssh_command "ps aux | grep -E 'python3.*mine|zion_cli.*mine' | grep -v grep")
    if [ "$mining_processes" != "N/A" ] && [ -n "$mining_processes" ]; then
        echo -e "${GREEN}✅ Běžící mining procesy:${NC}"
        echo "$mining_processes" | while read line; do
            echo -e "${CYAN}  $line${NC}"
        done
    else
        echo -e "${RED}❌ Žádné mining procesy neběží${NC}"
    fi
    
    echo ""
}

# Funkce pro zobrazení systémových statistik
show_system_stats() {
    echo -e "${PURPLE}🖥️  SYSTÉMOVÉ STATISTIKY${NC}"
    echo -e "${PURPLE}======================${NC}"
    
    # CPU usage
    local cpu_usage=$(ssh_command "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1")
    echo -e "${BLUE}🔥 CPU Usage: ${cpu_usage}%${NC}"
    
    # Memory usage
    local memory_info=$(ssh_command "free -h | awk 'NR==2{printf \"Used: %s/%s (%.2f%%)\", \$3, \$2, \$3/\$2*100}'")
    echo -e "${BLUE}💾 Memory: ${memory_info}${NC}"
    
    # Load average
    local load_avg=$(ssh_command "uptime | awk -F'load average:' '{print \$2}'")
    echo -e "${BLUE}📈 Load Average:${load_avg}${NC}"
    
    # Disk usage
    local disk_usage=$(ssh_command "df -h / | awk 'NR==2{printf \"Used: %s/%s (%s)\", \$3, \$2, \$5}'")
    echo -e "${BLUE}💿 Disk: ${disk_usage}${NC}"
    
    # Temperature (pokud je k dispozici)
    local temp=$(ssh_command "sensors 2>/dev/null | grep 'Core 0' | awk '{print \$3}' | head -1")
    if [ "$temp" != "N/A" ] && [ -n "$temp" ]; then
        echo -e "${BLUE}🌡️  CPU Temp: ${temp}${NC}"
    fi
    
    echo ""
}

# Funkce pro zobrazení GPU statistik
show_gpu_stats() {
    echo -e "${PURPLE}🎮 GPU STATISTIKY${NC}"
    echo -e "${PURPLE}===============${NC}"
    
    # NVIDIA GPU info
    local nvidia_info=$(ssh_command "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null")
    if [ "$nvidia_info" != "N/A" ] && [ -n "$nvidia_info" ]; then
        echo -e "${GREEN}✅ NVIDIA GPU:${NC}"
        echo "$nvidia_info" | while IFS=',' read name temp util mem_used mem_total; do
            echo -e "${CYAN}  🔥 $name: ${temp}°C | 📊 ${util}% | 💾 ${mem_used}MB/${mem_total}MB${NC}"
        done
    fi
    
    # AMD GPU info
    local amd_info=$(ssh_command "rocm-smi --showtemp --showuse --showmemuse 2>/dev/null | grep -E 'GPU|Temperature|GPU use|Memory use'")
    if [ "$amd_info" != "N/A" ] && [ -n "$amd_info" ]; then
        echo -e "${GREEN}✅ AMD GPU:${NC}"
        echo "$amd_info" | while read line; do
            echo -e "${CYAN}  $line${NC}"
        done
    fi
    
    if [ "$nvidia_info" = "N/A" ] && [ "$amd_info" = "N/A" ]; then
        echo -e "${YELLOW}⚠️  GPU info není k dispozici${NC}"
    fi
    
    echo ""
}

# Funkce pro zobrazení network statistik
show_network_stats() {
    echo -e "${PURPLE}🌐 NETWORK STATISTIKY${NC}"
    echo -e "${PURPLE}===================${NC}"
    
    # Active connections
    local connections=$(ssh_command "netstat -an | grep :8080 | wc -l")
    echo -e "${BLUE}🔗 Aktivní spojení na port 8080: ${connections}${NC}"
    
    # Network usage
    local network_info=$(ssh_command "cat /proc/net/dev | grep eth0 | awk '{printf \"RX: %.2f MB | TX: %.2f MB\", \$2/1024/1024, \$10/1024/1024}'")
    if [ "$network_info" != "N/A" ]; then
        echo -e "${BLUE}📡 Network Usage: ${network_info}${NC}"
    fi
    
    echo ""
}

# Funkce pro zobrazení logů
show_recent_logs() {
    echo -e "${PURPLE}📋 POSLEDNÍ LOGY${NC}"
    echo -e "${PURPLE}===============${NC}"
    
    # Mining logs
    local mining_logs=$(ssh_command "cd ${ZION_DIR} && tail -5 mining.log 2>/dev/null")
    if [ "$mining_logs" != "N/A" ] && [ -n "$mining_logs" ]; then
        echo -e "${GREEN}✅ Mining logs (posledních 5 řádků):${NC}"
        echo "$mining_logs" | while read line; do
            echo -e "${CYAN}  $line${NC}"
        done
    else
        echo -e "${YELLOW}⚠️  Mining logs nejsou k dispozici${NC}"
    fi
    
    echo ""
    
    # GPU optimizer logs
    local gpu_logs=$(ssh_command "cd ${ZION_DIR}/mining && tail -3 gpu_optimizer.log 2>/dev/null")
    if [ "$gpu_logs" != "N/A" ] && [ -n "$gpu_logs" ]; then
        echo -e "${GREEN}✅ GPU Optimizer logs (posledních 3 řádků):${NC}"
        echo "$gpu_logs" | while read line; do
            echo -e "${CYAN}  $line${NC}"
        done
    fi
    
    echo ""
}

# Kontrola SSH připojení
check_ssh_connection() {
    if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${SSH_USER}@${SSH_SERVER}" echo "SSH OK" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  SSH připojení vyžaduje heslo${NC}"
        # Pro monitoring budeme pokračovat s interaktivním SSH
        return 0
    fi
}

# Hlavní monitoring smyčka
main_monitor() {
    echo -e "${GREEN}🚀 Spouštění SSH Mining Monitoru...${NC}"
    echo -e "${YELLOW}💡 Pro ukončení stiskněte Ctrl+C${NC}"
    sleep 2
    
    while true; do
        clear_screen
        
        # Kontrola připojení
        check_ssh_connection
        
        # Zobrazení všech statistik
        show_mining_stats
        show_system_stats
        show_gpu_stats
        show_network_stats
        show_recent_logs
        
        # Užitečné příkazy
        echo -e "${CYAN}🛠️  UŽITEČNÉ PŘÍKAZY:${NC}"
        echo -e "${YELLOW}  Ctrl+C - Ukončit monitor${NC}"
        echo -e "${YELLOW}  ssh ${SSH_USER}@${SSH_SERVER} - Připojit k serveru${NC}"
        echo -e "${YELLOW}  ssh ${SSH_USER}@${SSH_SERVER} 'pkill -f zion' - Zastavit těžbu${NC}"
        
        # Čekání na další update
        echo ""
        echo -e "${BLUE}🔄 Další update za ${UPDATE_INTERVAL} sekund...${NC}"
        
        sleep $UPDATE_INTERVAL
    done
}

# Jednoduché menu
show_menu() {
    clear
    echo -e "${CYAN}🚀 ZION 2.7.1 SSH Mining Monitor${NC}"
    echo -e "${CYAN}=================================${NC}"
    echo ""
    echo -e "${YELLOW}Vyberte možnost:${NC}"
    echo -e "${GREEN}1) 📊 Spustit real-time monitoring${NC}"
    echo -e "${GREEN}2) 📈 Zobrazit aktuální stav (jednorázově)${NC}"
    echo -e "${GREEN}3) 📋 Zobrazit pouze mining logy${NC}"
    echo -e "${GREEN}4) 🔧 Spustit rychlou diagnostiku${NC}"
    echo -e "${GREEN}5) 🚪 Ukončit${NC}"
    echo ""
    read -p "Vaše volba [1-5]: " choice
    
    case $choice in
        1)
            main_monitor
            ;;
        2)
            clear_screen
            show_mining_stats
            show_system_stats
            show_gpu_stats
            echo -e "${YELLOW}Stiskněte Enter pro návrat do menu...${NC}"
            read
            show_menu
            ;;
        3)
            clear
            echo -e "${CYAN}📋 Mining Logy (posledních 50 řádků)${NC}"
            echo -e "${CYAN}===================================${NC}"
            ssh "${SSH_USER}@${SSH_SERVER}" "cd ${ZION_DIR} && tail -50 mining.log 2>/dev/null || echo 'Mining logy nejsou k dispozici'"
            echo ""
            echo -e "${YELLOW}Stiskněte Enter pro návrat do menu...${NC}"
            read
            show_menu
            ;;
        4)
            clear
            echo -e "${CYAN}🔧 Rychlá Diagnostika${NC}"
            echo -e "${CYAN}===================${NC}"
            
            echo -e "${BLUE}🔍 Kontrola SSH připojení...${NC}"
            check_ssh_connection && echo -e "${GREEN}✅ SSH OK${NC}"
            
            echo -e "${BLUE}🔍 Kontrola ZION procesů...${NC}"
            local processes=$(ssh_command "ps aux | grep -E 'python3|zion' | grep -v grep")
            if [ "$processes" != "N/A" ] && [ -n "$processes" ]; then
                echo -e "${GREEN}✅ ZION procesy běží${NC}"
            else
                echo -e "${RED}❌ Žádné ZION procesy${NC}"
            fi
            
            echo -e "${BLUE}🔍 Kontrola portů...${NC}"
            local port_check=$(ssh_command "netstat -ln | grep :8080")
            if [ "$port_check" != "N/A" ] && [ -n "$port_check" ]; then
                echo -e "${GREEN}✅ Port 8080 je otevřený${NC}"
            else
                echo -e "${YELLOW}⚠️  Port 8080 není aktivní${NC}"
            fi
            
            echo ""
            echo -e "${YELLOW}Stiskněte Enter pro návrat do menu...${NC}"
            read
            show_menu
            ;;
        5)
            echo -e "${GREEN}👋 Nashledanou!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Neplatná volba!${NC}"
            sleep 1
            show_menu
            ;;
    esac
}

# Spuštění
if [ "$1" = "--auto" ]; then
    main_monitor
else
    show_menu
fi