# denno-beta — Anthropic互換 LLM サーバー (CPU 推論)
# ビルド: docker compose build
# 起動:   docker compose up -d
FROM rust:1-slim AS build
WORKDIR /src
# tokenizers 等の依存が C/C++ コンパイラを必要とする
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# 依存のビルドキャッシュを効かせるため、まずマニフェストだけコピー
COPY Cargo.toml Cargo.lock .cargo/config.toml ./
RUN mkdir -p crates/denno-cli/src crates/denno-engine/src crates/denno-server/src \
    && echo "" > crates/denno-cli/src/main.rs \
    && echo "" > crates/denno-engine/src/lib.rs \
    && echo "" > crates/denno-server/src/lib.rs \
    && cargo build --release 2>/dev/null || true
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/denno /usr/local/bin/denno
# モデルキャッシュは $HOME/.cache/denno/hub → ボリューム /data に永続化
ENV HOME=/data
VOLUME /data
EXPOSE 8080
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["Qwen/Qwen2.5-0.5B-Instruct"]
