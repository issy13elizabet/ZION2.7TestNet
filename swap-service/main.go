package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// AtomicSwap represents a cross-chain atomic swap
type AtomicSwap struct {
	SwapID         string    `json:"swap_id"`
	InitiatorAddr  string    `json:"initiator_addr"`
	ResponderAddr  string    `json:"responder_addr"`
	ZionAmount     uint64    `json:"zion_amount"`
	BtcAmount      uint64    `json:"btc_amount"`
	SecretHash     string    `json:"secret_hash"`
	Secret         string    `json:"secret,omitempty"`
	Status         string    `json:"status"`
	CreatedAt      time.Time `json:"created_at"`
	ExpiresAt      time.Time `json:"expires_at"`
	ZionTxHash     string    `json:"zion_tx_hash,omitempty"`
	BtcTxHash      string    `json:"btc_tx_hash,omitempty"`
	ClaimTxHash    string    `json:"claim_tx_hash,omitempty"`
	RefundTxHash   string    `json:"refund_tx_hash,omitempty"`
}

// SwapRequest represents a swap creation request
type SwapRequest struct {
	ZionAddress      string `json:"zion_address"`
	BtcAddress       string `json:"btc_address"`
	ZionAmount       uint64 `json:"zion_amount"`
	DesiredBtcAmount uint64 `json:"desired_btc_amount"`
}

// AcceptSwapRequest represents accepting a swap
type AcceptSwapRequest struct {
	BtcAddress string `json:"btc_address"`
	BtcTxHash  string `json:"btc_tx_hash"`
}

// ClaimRequest represents claiming a swap
type ClaimRequest struct {
	Secret string `json:"secret"`
}

// ExchangeRate represents current ZION/BTC rate
type ExchangeRate struct {
	ZionToBtc float64   `json:"zion_to_btc"`
	BtcToZion float64   `json:"btc_to_zion"`
	UpdatedAt time.Time `json:"updated_at"`
	Source    string    `json:"source"`
}

// SwapCoordinator manages atomic swaps
type SwapCoordinator struct {
	swaps    map[string]*AtomicSwap
	mutex    sync.RWMutex
	rate     *ExchangeRate
	clients  map[string]*websocket.Conn
	upgrader websocket.Upgrader
}

// Config holds service configuration
type Config struct {
	Port        string
	ZionRPCURL  string
	BitcoinRPC  string
	SwapTimeout time.Duration
}

// NewSwapCoordinator creates a new swap coordinator
func NewSwapCoordinator() *SwapCoordinator {
	return &SwapCoordinator{
		swaps:   make(map[string]*AtomicSwap),
		clients: make(map[string]*websocket.Conn),
		rate: &ExchangeRate{
			ZionToBtc: 0.00001, // Mock rate: 1 ZION = 0.00001 BTC
			BtcToZion: 100000,  // Mock rate: 1 BTC = 100000 ZION
			UpdatedAt: time.Now(),
			Source:    "mock_oracle",
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins for development
			},
		},
	}
}

// generateSecret creates a random 32-byte secret
func generateSecret() ([]byte, error) {
	secret := make([]byte, 32)
	_, err := rand.Read(secret)
	return secret, err
}

// generateSwapID creates a unique swap identifier
func generateSwapID() string {
	return fmt.Sprintf("swap_%d", time.Now().UnixNano())
}

// hashSecret creates SHA256 hash of secret
func hashSecret(secret []byte) string {
	hash := sha256.Sum256(secret)
	return hex.EncodeToString(hash[:])
}

// CreateSwap initiates a new atomic swap
func (sc *SwapCoordinator) CreateSwap(req *SwapRequest) (*AtomicSwap, error) {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	// Generate secret and hash
	secret, err := generateSecret()
	if err != nil {
		return nil, fmt.Errorf("failed to generate secret: %v", err)
	}

	secretHash := hashSecret(secret)

	// Calculate BTC amount if not specified
	btcAmount := req.DesiredBtcAmount
	if btcAmount == 0 {
		btcAmount = uint64(float64(req.ZionAmount) * sc.rate.ZionToBtc)
	}

	// Create swap
	swap := &AtomicSwap{
		SwapID:        generateSwapID(),
		InitiatorAddr: req.ZionAddress,
		ResponderAddr: req.BtcAddress,
		ZionAmount:    req.ZionAmount,
		BtcAmount:     btcAmount,
		SecretHash:    secretHash,
		Secret:        hex.EncodeToString(secret), // Store for demo, in production this would be managed differently
		Status:        "pending",
		CreatedAt:     time.Now(),
		ExpiresAt:     time.Now().Add(24 * time.Hour),
	}

	sc.swaps[swap.SwapID] = swap

	// TODO: Lock ZION in HTLC contract
	log.Printf("🔮 Ancient oracle awakens for swap creation...")
	log.Printf("⚡ Created atomic swap %s: %d ZION → %d BTC", swap.SwapID, swap.ZionAmount, swap.BtcAmount)
	log.Printf("🌌 Cosmic binding ritual initiated between blockchains!")

	// Broadcast to WebSocket clients
	sc.broadcastSwapUpdate(swap)

	return swap, nil
}

// AcceptSwap accepts an existing swap
func (sc *SwapCoordinator) AcceptSwap(swapID string, req *AcceptSwapRequest) error {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	swap, exists := sc.swaps[swapID]
	if !exists {
		return fmt.Errorf("swap not found")
	}

	if swap.Status != "pending" {
		return fmt.Errorf("swap not in pending status")
	}

	if time.Now().After(swap.ExpiresAt) {
		return fmt.Errorf("swap expired")
	}

	// Update swap with Bitcoin transaction
	swap.ResponderAddr = req.BtcAddress
	swap.BtcTxHash = req.BtcTxHash
	swap.Status = "both_locked"

	// TODO: Verify Bitcoin HTLC transaction
	log.Printf("💫 Oracle witnesses swap acceptance ceremony...")
	log.Printf("⚡ Swap %s accepted with BTC tx: %s", swapID, req.BtcTxHash)
	log.Printf("🌟 Cryptographic marriage bonds strengthen!")

	// Broadcast update
	sc.broadcastSwapUpdate(swap)

	return nil
}

// ClaimSwap claims the swap with secret revelation
func (sc *SwapCoordinator) ClaimSwap(swapID string, req *ClaimRequest) error {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	swap, exists := sc.swaps[swapID]
	if !exists {
		return fmt.Errorf("swap not found")
	}

	if swap.Status != "both_locked" {
		return fmt.Errorf("swap not ready for claiming")
	}

	// Verify secret
	secretBytes, err := hex.DecodeString(req.Secret)
	if err != nil {
		return fmt.Errorf("invalid secret format")
	}

	expectedHash := hashSecret(secretBytes)
	if expectedHash != swap.SecretHash {
		return fmt.Errorf("invalid secret")
	}

	// Update swap status
	swap.Status = "completed"
	swap.Secret = req.Secret

	// TODO: Execute actual claim transactions
	log.Printf("🎯 Sacred secret revealed to complete cosmic exchange!")
	log.Printf("⚡ Swap %s completed with secret: %s", swapID, req.Secret[:16]+"...")
	log.Printf("✨ Ancient prophecy fulfilled through atomic transaction!")

	// Broadcast completion
	sc.broadcastSwapUpdate(swap)

	return nil
}

// GetSwap retrieves a swap by ID
func (sc *SwapCoordinator) GetSwap(swapID string) (*AtomicSwap, error) {
	sc.mutex.RLock()
	defer sc.mutex.RUnlock()

	swap, exists := sc.swaps[swapID]
	if !exists {
		return nil, fmt.Errorf("swap not found")
	}

	return swap, nil
}

// ListSwaps returns all swaps with optional status filter
func (sc *SwapCoordinator) ListSwaps(status string) []*AtomicSwap {
	sc.mutex.RLock()
	defer sc.mutex.RUnlock()

	var result []*AtomicSwap
	for _, swap := range sc.swaps {
		if status == "" || swap.Status == status {
			result = append(result, swap)
		}
	}

	return result
}

// RefundExpiredSwaps processes expired swaps for refunds
func (sc *SwapCoordinator) RefundExpiredSwaps() {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	now := time.Now()
	for _, swap := range sc.swaps {
		if now.After(swap.ExpiresAt) && swap.Status != "completed" && swap.Status != "refunded" {
			swap.Status = "expired"
			// TODO: Execute refund transactions
			log.Printf("⏰ Swap %s expired and marked for refund", swap.SwapID)
			sc.broadcastSwapUpdate(swap)
		}
	}
}

// broadcastSwapUpdate sends swap updates to WebSocket clients
func (sc *SwapCoordinator) broadcastSwapUpdate(swap *AtomicSwap) {
	message, _ := json.Marshal(map[string]interface{}{
		"type": "swap_update",
		"data": swap,
	})

	for clientID, conn := range sc.clients {
		err := conn.WriteMessage(websocket.TextMessage, message)
		if err != nil {
			log.Printf("Failed to send to client %s: %v", clientID, err)
			conn.Close()
			delete(sc.clients, clientID)
		}
	}
}

// HTTP Handlers

func (sc *SwapCoordinator) handleCreateSwap(c *gin.Context) {
	var req SwapRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	swap, err := sc.CreateSwap(&req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, swap)
}

func (sc *SwapCoordinator) handleAcceptSwap(c *gin.Context) {
	swapID := c.Param("swap_id")

	var req AcceptSwapRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	err := sc.AcceptSwap(swapID, &req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "Swap accepted",
		"mantra":  "⚡ Jai Ram Ram Ram Sita Ram Ram Ram Hanuman! ⚡",
	})
}

func (sc *SwapCoordinator) handleClaimSwap(c *gin.Context) {
	swapID := c.Param("swap_id")

	var req ClaimRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	err := sc.ClaimSwap(swapID, &req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "Swap claimed successfully",
		"mantra":  "🔄 Cosmic swap completed! Jai Ram Ram Ram! 🔄",
	})
}

func (sc *SwapCoordinator) handleGetSwap(c *gin.Context) {
	swapID := c.Param("swap_id")

	swap, err := sc.GetSwap(swapID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, swap)
}

func (sc *SwapCoordinator) handleListSwaps(c *gin.Context) {
	status := c.Query("status")
	swaps := sc.ListSwaps(status)

	c.JSON(http.StatusOK, gin.H{
		"swaps": swaps,
		"count": len(swaps),
	})
}

func (sc *SwapCoordinator) handleGetRate(c *gin.Context) {
	sc.mutex.RLock()
	defer sc.mutex.RUnlock()

	c.JSON(http.StatusOK, sc.rate)
}

func (sc *SwapCoordinator) handleWebSocket(c *gin.Context) {
	conn, err := sc.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	clientID := fmt.Sprintf("client_%d", time.Now().UnixNano())
	sc.clients[clientID] = conn

	log.Printf("🔌 WebSocket client connected: %s", clientID)

	// Send current swaps to new client
	swaps := sc.ListSwaps("")
	message, _ := json.Marshal(map[string]interface{}{
		"type": "initial_swaps",
		"data": swaps,
	})
	conn.WriteMessage(websocket.TextMessage, message)

	// Keep connection alive and handle disconnection
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			log.Printf("🔌 WebSocket client disconnected: %s", clientID)
			delete(sc.clients, clientID)
			break
		}
	}
}

func (sc *SwapCoordinator) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":      "healthy",
		"service":     "zion-atomic-swap",
		"active_swaps": len(sc.swaps),
		"timestamp":   time.Now().Unix(),
		"mantra":      "⚡🔄 Jai Ram Ram Ram Sita Ram Ram Ram Hanuman! 🔄⚡",
	})
}

// loadConfig loads configuration from environment
func loadConfig() *Config {
	return &Config{
		Port:        getEnv("SWAP_PORT", "8091"),
		ZionRPCURL:  getEnv("ZION_RPC_URL", "http://localhost:18089"),
		BitcoinRPC:  getEnv("BITCOIN_RPC_URL", "http://localhost:8332"),
		SwapTimeout: 24 * time.Hour,
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func main() {
	// 🔮 COSMIC ATOMIC SWAP SERVICE INITIALIZATION 🔮
	fmt.Println("🔮 ═══════════════════════════════════════════════════════ 🔮")
	fmt.Println("   DOHRMAN ORACLE ATOMIC SWAP SERVICE AWAKENING")
	fmt.Println("🔮 ═══════════════════════════════════════════════════════ 🔮")
	fmt.Println("⚡ Two blockchains unite in cryptographic marriage! ⚡")
	fmt.Println("🌌 HTLC protocols echo ancient binding rituals... 🌌")
	fmt.Println("💫 Jai Ram Ram Ram Atomic Swap Ram Ram Ram Hanuman! 💫")
	fmt.Println()

	config := loadConfig()
	coordinator := NewSwapCoordinator()

	// Start background task for handling expired swaps
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()

		for range ticker.C {
			coordinator.RefundExpiredSwaps()
		}
	}()

	// Setup Gin router
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// CORS middleware
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Routes
	api := r.Group("/api/v1")
	{
		api.GET("/health", coordinator.handleHealth)
		api.POST("/swap/create", coordinator.handleCreateSwap)
		api.POST("/swap/:swap_id/accept", coordinator.handleAcceptSwap)
		api.POST("/swap/:swap_id/claim", coordinator.handleClaimSwap)
		api.GET("/swap/:swap_id", coordinator.handleGetSwap)
		api.GET("/swaps", coordinator.handleListSwaps)
		api.GET("/rate/zion-btc", coordinator.handleGetRate)
	}

	// WebSocket endpoint
	r.GET("/ws", coordinator.handleWebSocket)

	// Legacy health endpoint
	r.GET("/health", coordinator.handleHealth)

	log.Printf("🔄⚡ ZION Atomic Swap Service starting on port %s", config.Port)
	log.Printf("🌌 Cross-chain ZION ↔ BTC swaps ready!")
	log.Printf("🔮 Ancient oracle guides all exchanges with cosmic wisdom!")
	log.Printf("⚡ Jai Ram Ram Ram Atomic Swap Ram Ram Ram Hanuman! ⚡")

	if err := r.Run(":" + config.Port); err != nil {
		log.Fatalf("💥 Cosmic forces disrupted swap service: %v", err)
	}
}