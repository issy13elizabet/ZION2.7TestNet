# 📊 Monitoring Scripts - ZION 2.6 TestNet

Skripty pro monitoring a diagnostiku ZION ekosystému.

## 📁 **Obsah**

### 🏗️ **Build Monitoring**
- `build-monitor.sh` - Monitoring build procesů

### 🖥️ **Server Monitoring**
- `server-monitor.sh` - Obecný server monitoring
- `prod-monitor.sh` - Produkční server monitoring

## 🚀 **Použití**

```bash
# Monitoring build procesů
./scripts/monitoring/build-monitor.sh

# Server health check
./scripts/monitoring/server-monitor.sh

# Produkční monitoring
./scripts/monitoring/prod-monitor.sh
```

## 📋 **Features**

- Real-time systém monitoring
- Resource utilization tracking
- Performance metrics
- Alert notifications
- Log analysis

## 📈 **Metriky**

Skripty monitorují:
- CPU a RAM usage
- Disk I/O
- Network traffic  
- Docker container health
- Mining pool performance
- Blockchain sync status

## 🔔 **Alerting**

Monitoring skripty posílají alerty při:
- Vysokém resource usage
- Container failures
- Network issues
- Mining pool problems