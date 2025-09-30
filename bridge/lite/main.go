package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
    "time"
    "bytes"
    "io"

    "github.com/gin-gonic/gin"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func getEnv(key, def string) string {
    if v := os.Getenv(key); v != "" { return v }
    return def
}

func main() {
    port := getEnv("BRIDGE_PORT", "8090")
    daemonURL := getEnv("DAEMON_RPC_URL", "http://legacy-daemon:18081")
    horizon := getEnv("STELLAR_HORIZON_URL", "https://horizon.stellar.org")

    reg := prometheus.NewRegistry()
    reg.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
    reg.MustRegister(prometheus.NewGoCollector())

    gin.SetMode(gin.ReleaseMode)
    r := gin.Default()

    // CORS
    r.Use(func(c *gin.Context) {
        c.Header("Access-Control-Allow-Origin", "*")
        c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if c.Request.Method == "OPTIONS" { c.AbortWithStatus(204); return }
        c.Next()
    })

    // Health
    r.GET("/health", func(c *gin.Context){
        c.JSON(http.StatusOK, gin.H{"status":"healthy","service":"zion-go-bridge-lite","ts": time.Now().Unix()})
    })

    // Metrics
    r.GET("/metrics", gin.WrapH(promhttp.HandlerFor(reg, promhttp.HandlerOpts{})))

    // API
    api := r.Group("/api/v1")
    {
        api.GET("/health", func(c *gin.Context){
            c.JSON(http.StatusOK, gin.H{"ok":true, "ts": time.Now().Unix()})
        })
        // Minimal proxies
        api.GET("/daemon/get_info", func(c *gin.Context){
            url := fmt.Sprintf("%s/json_rpc", daemonURL)
            payload := []byte(`{"jsonrpc":"2.0","id":1,"method":"get_info"}`)
            req, _ := http.NewRequest("POST", url, bytes.NewReader(payload))
            req.Header.Set("Content-Type","application/json")
            client := &http.Client{ Timeout: 5 * time.Second }
            resp, err := client.Do(req)
            if err != nil { c.JSON(http.StatusBadGateway, gin.H{"error":"daemon_unreachable"}); return }
            defer resp.Body.Close()
            b, _ := io.ReadAll(resp.Body)
            c.Data(resp.StatusCode, "application/json", b)
        })
        api.GET("/stellar/ledger", func(c *gin.Context){
            url := fmt.Sprintf("%s/ledgers?order=desc&limit=1", horizon)
            resp, err := http.Get(url)
            if err != nil { c.JSON(http.StatusBadGateway, gin.H{"error":"horizon_unreachable"}); return }
            defer resp.Body.Close()
            b, _ := io.ReadAll(resp.Body)
            c.Data(resp.StatusCode, "application/json", b)
        })
    }

    log.Printf("ZION Go Bridge (lite) on :%s", port)
    if err := r.Run(":"+port); err != nil { log.Fatal(err) }
}
