FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install --production

# Copy built TypeScript files
COPY dist ./dist
COPY src ./src
COPY modules ./modules
COPY server.js ./

# Environment variables
ENV PORT=8080
ENV NODE_ENV=production

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the server
CMD ["node", "server.js"]