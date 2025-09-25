FROM golang:1.21-alpine AS builder

# Jai Ram Ram Ram Sita Ram Ram Ram Hanuman!
LABEL maintainer="ZION Atomic Swap Team"
LABEL description="ZION ↔ BTC Atomic Swap Service"

# Install dependencies
RUN apk add --no-cache git build-base

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o zion-atomic-swap .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates curl

WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/zion-atomic-swap .

# Expose ports
EXPOSE 8091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8091/health || exit 1

# Run the application
CMD ["./zion-atomic-swap"]