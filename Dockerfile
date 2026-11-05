# syntax=docker/dockerfile:1
ARG MODEL=isnet-anime
FROM golang:1.25-bookworm AS build
ARG TARGETARCH
ARG MODEL
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY go.mod go.sum ./
RUN go mod download && go mod verify
# Match the native C API version used by onnxruntime_go v1.22.0.
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in amd64) arch=x64 ;; arm64) arch=aarch64 ;; *) exit 1 ;; esac; \
    curl -fL --retry 5 "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-${arch}-1.22.0.tgz" -o /tmp/ort.tgz; \
    mkdir -p /opt/ort; tar -xzf /tmp/ort.tgz --strip-components=1 -C /opt/ort; rm /tmp/ort.tgz
# Upstream rembg publishes these integrity checks for its model releases.
RUN set -eux; \
    case "$MODEL" in u2net) filename=u2net.onnx; checksum=60024c5c889badc19c04ad937298a77b ;; u2netp) filename=u2netp.onnx; checksum=8e83ca70e441ab06c318d82300c84806 ;; isnet-general-use) filename=isnet-general-use.onnx; checksum=fc16ebd8b0c10d971d3513d564d01e29 ;; isnet-anime) filename=isnet-anime.onnx; checksum=6f184e756bb3bd901c8849220a83e38e ;; *) exit 1 ;; esac; \
    mkdir -p /opt/models; \
    curl -fL --retry 5 "https://github.com/danielgatis/rembg/releases/download/v0.0.0/${filename}" -o /opt/models/model.onnx; \
    echo "$checksum  /opt/models/model.onnx" | md5sum -c -
COPY . .
RUN CGO_ENABLED=1 go test ./... && go vet ./... && CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/cdrembg .

FROM debian:bookworm-slim AS runtime
ARG MODEL
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/cdrembg /app/cdrembg
COPY --from=build /opt/ort/lib/ /app/lib/
COPY --from=build /opt/ort/LICENSE /app/licenses/onnxruntime-LICENSE
COPY --from=build /opt/ort/ThirdPartyNotices.txt /app/licenses/onnxruntime-ThirdPartyNotices.txt
COPY --from=build /opt/models/ /app/models/
ENV PORT=8080 MODEL_PATH=/app/models/model.onnx ONNX_RUNTIME_LIBRARY_PATH=/app/lib/libonnxruntime.so.1.22.0 ORT_THREADS=1 GOMEMLIMIT=128MiB MALLOC_ARENA_MAX=2
ENV MODEL_NAME=${MODEL}
USER 65532:65532
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1
ENTRYPOINT ["/app/cdrembg"]
