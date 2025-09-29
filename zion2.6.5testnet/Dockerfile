# 🌟 ZION PRODUCTION SERVER
# Pure JavaScript, No TypeScript Compilation Issues

FROM node:18-alpine

# Install basic utilities
RUN apk add --no-cache curl

# Security updates
RUN apk upgrade

# Create app directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install --production

# Copy application code
COPY server.js ./

# Create non-root user
RUN addgroup -g 1001 -S zion && \
    adduser -S zion -u 1001

# Change ownership
RUN chown -R zion:zion /app

# Switch to non-root user
USER zion

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8888/health || exit 1

# Start command
CMD ["node", "server.js"]