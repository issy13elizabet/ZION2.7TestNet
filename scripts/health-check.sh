#!/bin/bash
# ZION Health Check Script
# Automated health monitoring for ZION TestNet infrastructure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ZION_CORE_URL="http://localhost:8888"
WALLET_ADAPTER_URL="http://localhost:3001"
POOL_URL="http://localhost:3333"
PROMETHEUS_URL="http://localhost:9090"

# Health check results
HEALTH_STATUS=0
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo -e "${BLUE}🔍 ZION TestNet Health Check - ${TIMESTAMP}${NC}"
echo "================================================="

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking ${service_name}... "
    
    if timeout 10 curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✅ HEALTHY${NC}"
        return 0
    else
        echo -e "${RED}❌ UNHEALTHY${NC}"
        HEALTH_STATUS=1
        return 1
    fi
}

# Function to check Docker container status
check_container() {
    local container_name=$1
    echo -n "Checking container ${container_name}... "
    
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        local status=$(docker inspect --format="{{.State.Health.Status}}" "$container_name" 2>/dev/null)
        if [ "$status" = "healthy" ] || [ "$status" = "" ]; then
            echo -e "${GREEN}✅ RUNNING${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  UNHEALTHY (${status})${NC}"
            HEALTH_STATUS=1
            return 1
        fi
    else
        echo -e "${RED}❌ NOT RUNNING${NC}"
        HEALTH_STATUS=1
        return 1
    fi
}

# Function to check network connectivity
check_network_connectivity() {
    echo -n "Checking network connectivity... "
    
    if ping -c 1 -W 3 8.8.8.8 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ CONNECTED${NC}"
        return 0
    else
        echo -e "${RED}❌ NO INTERNET${NC}"
        HEALTH_STATUS=1
        return 1
    fi
}

# Function to check disk space
check_disk_space() {
    echo -n "Checking disk space... "
    
    local usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$usage" -lt 90 ]; then
        echo -e "${GREEN}✅ ${usage}% used${NC}"
        return 0
    else
        echo -e "${RED}❌ ${usage}% used (HIGH)${NC}"
        HEALTH_STATUS=1
        return 1
    fi
}

# Function to check system load
check_system_load() {
    echo -n "Checking system load... "
    
    local load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//' | sed 's/ //')
    local cpu_count=$(nproc)
    
    # Check if load is a valid number
    if [[ "$load" =~ ^[0-9]*\.?[0-9]+$ ]] && [ -n "$load" ]; then
        local load_percentage=$(echo "scale=0; $load * 100 / $cpu_count" | bc -l 2>/dev/null || echo "0")
        
        if [ "$load_percentage" -lt 80 ]; then
            echo -e "${GREEN}✅ ${load} (${load_percentage}%)${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  ${load} (${load_percentage}%)${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}⚠️  Unable to parse load${NC}"
        return 0
    fi
}

echo -e "\n${BLUE}🐳 Docker Containers:${NC}"
check_container "zion-core"
check_container "pool-nodejs"
check_container "wallet-adapter"
check_container "zion-rpc-shim"

echo -e "\n${BLUE}🌐 Service Endpoints:${NC}"
check_service "ZION Core API" "$ZION_CORE_URL/health"
check_service "Wallet Adapter" "$WALLET_ADAPTER_URL/healthz"
check_service "Mining Pool (Stratum info)" "$POOL_URL" "404"

echo -e "\n${BLUE}🔧 System Resources:${NC}"
check_network_connectivity
check_disk_space  
check_system_load

echo -e "\n${BLUE}📊 Docker Network Status:${NC}"
echo -n "Checking zion-seeds network... "
if docker network ls | grep -q "zion-seeds"; then
    echo -e "${GREEN}✅ EXISTS${NC}"
else
    echo -e "${RED}❌ MISSING${NC}"
    HEALTH_STATUS=1
fi

# Memory usage check
check_memory_usage() {
    echo -n "Checking memory usage... "
    local mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ -n "$mem_usage" ] && [ "$mem_usage" -lt 90 ]; then
        echo -e "${GREEN}✅ ${mem_usage}% used${NC}"
    elif [ -n "$mem_usage" ]; then
        echo -e "${YELLOW}⚠️  ${mem_usage}% used (HIGH)${NC}"
    else
        echo -e "${YELLOW}⚠️  Unable to get memory usage${NC}"
    fi
}

check_memory_usage

echo "================================================="

if [ $HEALTH_STATUS -eq 0 ]; then
    echo -e "${GREEN}🎉 All systems operational!${NC}"
    echo -e "${GREEN}ZION TestNet is healthy and running smoothly.${NC}"
else
    echo -e "${RED}⚠️  Issues detected!${NC}"
    echo -e "${YELLOW}Some services need attention. Check the details above.${NC}"
fi

echo -e "\n${BLUE}📋 Additional Info:${NC}"
echo "- Last check: $TIMESTAMP"
echo "- Docker containers: $(docker ps --format 'table {{.Names}}' | tail -n +2 | wc -l) running"
echo "- System uptime: $(uptime -p)"

# Save health status to file for monitoring
echo "$TIMESTAMP,$HEALTH_STATUS" >> /tmp/zion-health.log

exit $HEALTH_STATUS