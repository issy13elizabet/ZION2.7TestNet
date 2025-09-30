FROM golang:1.21-alpine AS builder

# Jai Ram Ram Ram Sita Ram Ram Ram Hanuman!
LABEL maintainer="ZION Lightning Team"
LABEL description="ZION Lightning Network Bridge Service"

# Install dependencies
RUN apk add --no-cache git build-base

# Set working directory
WORKDIR /app

# Build isolated lite module to avoid pulling optional heavy deps
WORKDIR /app/lite
COPY lite/go.mod ./
RUN go get github.com/gin-gonic/gin@v1.9.1 && \
  go get github.com/prometheus/client_golang@v1.17.0 && \
  go mod download

# Copy only lite module source
COPY lite/ .

# Ensure module files are consistent and go.sum is generated
RUN go mod tidy

# Build the lite bridge (without LND)
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /app/zion-lightning-bridge .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates curl

WORKDIR /app

# Add non-root user
RUN addgroup -S zion && adduser -S zion -G zion

# Copy the binary from builder stage
COPY --from=builder /app/zion-lightning-bridge .

# Expose port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8090/api/v1/health || exit 1

# Run the application
USER zion
CMD ["./zion-lightning-bridge"]