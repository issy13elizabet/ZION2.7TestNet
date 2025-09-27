# syntax=docker/dockerfile:1

FROM ubuntu:22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git libssl-dev libleveldb-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j $(nproc)

FROM ubuntu:22.04 AS runtime
RUN apt-get update && apt-get install -y \
    libssl3 leveldb && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /zion
COPY --from=build /app/build/ziond /usr/local/bin/ziond
COPY --from=build /app/build/zion_wallet /usr/local/bin/zion_wallet
COPY --from=build /app/build/zion_miner /usr/local/bin/zion_miner

# default data/config locations inside container
RUN mkdir -p /zion/data /zion/config /zion/logs

EXPOSE 18080 18081
ENTRYPOINT ["/usr/local/bin/ziond"]
CMD ["--config=/zion/config/zion.conf", "--datadir=/zion/data"]
