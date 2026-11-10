# syntax=docker/dockerfile:1
ARG MODEL=isnet-general-use

FROM golang:1.25-bookworm AS build
ARG TARGETARCH
ARG MODEL
WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY go.mod go.sum ./
RUN go mod download && go mod verify

# Fetch ONNX Runtime 1.22.0 matching onnxruntime_go v1.22.0 C API and verify SHA-256.
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
      amd64)   arch=x64;     ort_sha256=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3 ;; \
      arm64)   arch=aarch64; ort_sha256=bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702 ;; \
      *)       exit 1 ;; \
    esac; \
    curl -fL --retry 5 "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-${arch}-1.22.0.tgz" -o /tmp/ort.tgz; \
    echo "${ort_sha256}  /tmp/ort.tgz" | sha256sum -c -; \
    mkdir -p /opt/ort; \
    tar -xzf /tmp/ort.tgz --strip-components=1 -C /opt/ort; \
    rm /tmp/ort.tgz

# Fetch segmentation model weights at build time and verify SHA-256.
RUN set -eux; \
    case "$MODEL" in \
      isnet-general-use) filename=isnet-general-use.onnx; sha256=60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a ;; \
      isnet-anime)       filename=isnet-anime.onnx;       sha256=f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99 ;; \
      u2net)             filename=u2net.onnx;             sha256=8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491 ;; \
      u2netp)            filename=u2netp.onnx;            sha256=309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8 ;; \
      *)                 exit 1 ;; \
    esac; \
    mkdir -p /opt/models; \
    curl -fL --retry 5 "https://github.com/danielgatis/rembg/releases/download/v0.0.0/${filename}" -o /opt/models/model.onnx; \
    echo "${sha256}  /opt/models/model.onnx" | sha256sum -c -

COPY . .
RUN CGO_ENABLED=1 go test -race ./... \
    && CGO_ENABLED=1 go vet ./... \
    && CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/cdrembg .

FROM debian:bookworm-slim AS runtime
ARG MODEL
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /out/cdrembg /app/cdrembg
COPY --from=build /opt/ort/lib/ /app/lib/
COPY --from=build /opt/ort/LICENSE /app/licenses/onnxruntime-LICENSE
COPY --from=build /opt/ort/ThirdPartyNotices.txt /app/licenses/onnxruntime-ThirdPartyNotices.txt
COPY --from=build /opt/models/ /app/models/

ENV PORT=8080 \
    MODEL_PATH=/app/models/model.onnx \
    ONNX_RUNTIME_LIBRARY_PATH=/app/lib/libonnxruntime.so.1.22.0 \
    GOMEMLIMIT=256MiB \
    MALLOC_ARENA_MAX=2 \
    MODEL_NAME=${MODEL}

USER 65532:65532
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 CMD ["/app/cdrembg", "-healthcheck"]
ENTRYPOINT ["/app/cdrembg"]
