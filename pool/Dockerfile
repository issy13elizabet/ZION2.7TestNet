FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --production || npm install --production
COPY server.js .
ENV POOL_PORT=3333
ENV POOL_BIND=0.0.0.0
EXPOSE 3333
CMD ["npm", "start"]
