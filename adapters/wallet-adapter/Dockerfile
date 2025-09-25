FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm i --only=production || npm i --omit=dev
COPY . .
ENV ADAPTER_PORT=18099
EXPOSE 18099
CMD ["npm","start"]
