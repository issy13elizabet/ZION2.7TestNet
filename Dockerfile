# Multi-stage Dockerfile for Zion cryptocurrency daemon and pool server

# Build stage
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source code
COPY . .

# Clean any existing CMake cache and manually clone RandomX
RUN rm -f CMakeCache.txt && \
    rm -rf CMakeFiles/ && \
    rm -rf build/ && \
    rm -rf external/randomx && \
    mkdir -p external && \
    cd external && \
    git clone https://github.com/tevador/RandomX.git randomx && \
    cd randomx && \
    git checkout tags/v1.2.1 && \
    cd /build && \
    ls -la external/randomx/ && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies including gosu
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    netcat-openbsd \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash zion && \
    mkdir -p /home/zion/.zion

# Copy binaries from build stage
COPY --from=builder /build/build/ziond /usr/local/bin/
COPY --from=builder /build/build/zion_miner /usr/local/bin/
COPY --from=builder /build/build/zion_wallet /usr/local/bin/
COPY --from=builder /build/build/zion_genesis /usr/local/bin/

# Make binaries executable
RUN chmod +x /usr/local/bin/zion*

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Keep running as root; entrypoint will drop privileges to 'zion' via gosu
WORKDIR /home/zion

# Expose ports
EXPOSE 18080 18081 3333

# Health check: consider pool or daemon
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD bash -lc 'if [ "${ZION_MODE:-daemon}" = "pool" ]; then nc -z 127.0.0.1 ${POOL_PORT:-3333}; else curl -fsS http://127.0.0.1:${RPC_PORT:-18081}/status; fi'

ENTRYPOINT ["/entrypoint.sh"]
CMD ["daemon"]