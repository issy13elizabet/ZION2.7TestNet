FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --production || npm install --production
COPY server.js .
ENV SHIM_PORT=18089
ENV ZION_RPC_URL=http://seed1:18081/json_rpc
EXPOSE 18089
CMD ["npm", "start"]
