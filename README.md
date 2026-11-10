# cdrembg

[![CI](https://github.com/cosmind-rusu/cdrembg/actions/workflows/ci.yml/badge.svg)](https://github.com/cosmind-rusu/cdrembg/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-17603e.svg)](LICENSE)

Self-hosted background removal service written in Go on top of CPU ONNX Runtime (`v1.22.0`) and IS-Net / U2-Net segmentation models. `cdrembg` ships as a single executable with an embedded web UI and a minimal REST API, processes uploads strictly in memory without touching disk, and returns full-resolution transparent PNGs.

## Architecture & Engineering Highlights

- **Single binary, zero external frontend dependencies:** HTML, CSS, and vanilla JavaScript are embedded via `embed.FS` and served under a strict `Content-Security-Policy` (`default-src 'none'`, no `'unsafe-inline'`, no third-party CDNs).
- **Zero-allocation steady-state pixel pipeline:** Direct `*image.NRGBA` slice loops, reusable resampling buffers, in-place alpha compositing, and `png.EncoderBufferPool` achieve `0 B/op` and `0 allocs/op` across both tensor preprocessing and mask compositing.
- **Anti-aliased resampling & EXIF normalization:** High-resolution inputs use separable box/area integration when downsampling to model dimensions (preventing stair-stepped contours on fine edges) and automatically normalize JPEG EXIF orientation (`1..8`) so portrait mobile photos render upright.
- **Bounded memory & explicit load shedding:** Header inspection (`image.DecodeConfig`) rejects invalid formats and oversized dimensions (>16 MP) before allocating pixel buffers. Network upload parsing is decoupled from the single-slot ONNX inference gate; concurrent inference bursts immediately receive `503 Service Unavailable` with `Retry-After: 5` instead of queueing toward an OOM kill.
- **Hardened, air-gapped container runtime:** Multi-arch (`linux/amd64`, `linux/arm64`) Docker builds pin and verify SHA-256 digests for both ONNX Runtime and model weights at build time. The runtime image drops `curl`, uses a built-in `-healthcheck` flag, runs as `65532:65532`, and supports `--read-only --cap-drop=ALL --security-opt=no-new-privileges`.
- **CGO-isolated package boundary:** ONNX Runtime bindings are isolated behind `//go:build cgo` in `internal/processor`, allowing unit tests and image pipeline benchmarks to run anywhere with `CGO_ENABLED=0`.

## Project Structure

```text
cdrembg/
├── main.go                         # HTTP server, two-stage admission control, graceful shutdown
├── main_test.go                    # Route, security header, admission, and upload limit tests
├── internal/processor/
│   ├── processor.go                # Model profiles, zero-alloc resampling, alpha compositing
│   ├── exif.go                     # Pure-Go JPEG EXIF orientation parser & fast NRGBA conversion
│   ├── ort_cgo.go                  # CGO ONNX Runtime session & preallocated NCHW tensors
│   ├── ort_nocgo.go                # Non-CGO build stub for portable unit testing
│   └── processor_test.go           # Pipeline unit tests, EXIF tests, and memory benchmarks
├── static/
│   ├── index.html                  # Embedded comparison UI
│   ├── styles.css                  # Responsive layout & WCAG AA palette
│   └── app.js                      # Drag/drop, clipboard paste, auto-retry on 503
├── Dockerfile                      # Multi-stage, SHA-256 verified amd64/arm64 container build
├── railway.json                    # Railway deployment & healthcheck configuration
└── .github/workflows/ci.yml        # Formatting, pure-Go tests, CGO race detector, E2E Docker test
```

## Quick Start with Docker

Both ONNX Runtime `1.22.0` and the selected segmentation model are fetched and SHA-256 verified at **build time**; container startup requires no outbound network access.

```sh
docker build -t cdrembg .
docker run --rm --name cdrembg \
  -p 8080:8080 \
  --memory=3g --memory-swap=3g \
  --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  cdrembg
```

Open `http://localhost:8080` to drop, select, or paste (`Ctrl+V`) a JPEG or PNG image.

### Model Selection

Select the baked-in model at build time with `--build-arg MODEL=<name>`:

| `MODEL` build arg | Resolution | Normalization (`mean` / `std`) | Best suited for |
|---|---|---|---|
| `isnet-general-use` *(default)* | `1024×1024` | `[0.5, 0.5, 0.5]` / `[1, 1, 1]` | General photographs, products, portraits |
| `isnet-anime` | `1024×1024` | `[0.485, 0.456, 0.406]` / `[1, 1, 1]` | Illustrations, anime, line art, characters |
| `u2net` | `320×320` | `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` | Faster CPU inference with lower contour detail |
| `u2netp` | `320×320` | `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` | Constrained memory hosts (~4.6 MB weights) |

```sh
# High-resolution general segmentation (default)
docker build -t cdrembg .

# Specialized profile for illustrations and anime
docker build --build-arg MODEL=isnet-anime -t cdrembg:anime .

# Lightweight 320x320 model for low-memory hosts
docker build --build-arg MODEL=u2netp -t cdrembg:small .
```

> **Note:** Docker stores the selected weights at `/app/models/model.onnx` and sets `MODEL_NAME` accordingly. Do not override `MODEL_NAME` at runtime without also mounting matching weights at `MODEL_PATH`.

## REST API

```sh
# Readiness probe
curl --fail-with-body http://localhost:8080/health

# Remove background and write transparent PNG
curl --fail-with-body -F 'file=@photo.jpg' http://localhost:8080/api/remove -o cutout.png
```

| Endpoint | Method | Response |
|---|---|---|
| `/` | `GET` | Embedded web application |
| `/health` | `GET` | `200 OK` · `{"status":"ok"}` |
| `/api/remove` | `POST` | `200 OK` · `image/png` (`X-Process-Time-Ms` header included) |

### Limits & Error Semantics

Uploads accept a single `file` field containing a valid **JPEG** or **PNG** up to **10 MiB** and **16 megapixels** (`16,000,000` decoded pixels). Content types and file extensions are never trusted; `image.DecodeConfig` validates magic bytes and dimensions before full pixel decoding. Existing alpha channels in source PNGs are preserved and multiplied by the predicted foreground mask.

All error responses return `application/json` (`{"error":"..."}`):

| Status | Meaning |
|---|---|
| `400 Bad Request` | Missing/malformed `multipart/form-data`, empty file, or multiple file fields |
| `413 Payload Too Large` | Upload exceeds 10 MiB |
| `422 Unprocessable Entity` | Unsupported image format, corrupt stream, or dimensions above 16 MP |
| `503 Service Unavailable` | Inference slot busy (`Retry-After: 5`); retry with backoff |
| `500 Internal Server Error` | Unexpected native runtime or encoding failure |

## Configuration

| Variable | Default (outside Docker) | Description |
|---|---|---|
| `PORT` | `8080` | TCP listen port (`1`–`65535`) |
| `MODEL_NAME` | `isnet-general-use` | `isnet-general-use`, `isnet-anime`, `u2net`, or `u2netp` |
| `MODEL_PATH` | `models/<MODEL_NAME>.onnx` | Path to ONNX model weights matching `MODEL_NAME` |
| `ONNX_RUNTIME_LIBRARY_PATH` | `lib/libonnxruntime.so.1.22.0` | Path to ONNX Runtime `1.22.0` shared library (`.so`, `.dylib`, `.dll`) |
| `ORT_THREADS` | `min(4, NumCPU)` | Intra-op CPU threads (`1`–`16`) |
| `GOMEMLIMIT` | Go runtime default | Soft Go heap target (`256MiB` in Docker; excludes native C++ allocations) |

## Local Development & Benchmarks

Unit tests and benchmarks run without CGO or native libraries:

```sh
CGO_ENABLED=0 go test -v ./...
CGO_ENABLED=0 go test -bench=. -benchmem ./internal/processor
```

Reference benchmark on a 4 MP (`2000×2000`) image with the `1024×1024` IS-Net profile (AMD Ryzen 9 5950X, Go 1.25.1):

```text
BenchmarkPreprocess4MP_ISNet-32       39    30964677 ns/op    0 B/op    0 allocs/op
BenchmarkApplyMask4MP_ISNet-32       127    10085070 ns/op    0 B/op    0 allocs/op
```

To run full native inference locally without Docker, install a C compiler and download [ONNX Runtime 1.22.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0) plus matching weights from [rembg releases](https://github.com/danielgatis/rembg/releases/tag/v0.0.0):

```sh
mkdir -p lib models
curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz -o /tmp/ort.tgz
tar -xzf /tmp/ort.tgz -C /tmp && cp -a /tmp/onnxruntime-linux-x64-1.22.0/lib/. lib/
curl -fL https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx -o models/isnet-general-use.onnx
CGO_ENABLED=1 go run .
```

On Windows, install MinGW-w64 GCC and point `ONNX_RUNTIME_LIBRARY_PATH` to `onnxruntime.dll`; on macOS, point it to `libonnxruntime.1.22.0.dylib`.

## Deploying on Railway

1. Push the repository to GitHub and create a Railway service from the repo (`railway.json` automatically selects the `Dockerfile` builder and `/health` readiness check).
2. Provision **2–3 GB RAM** for `1024×1024` IS-Net profiles (or **1 GB** for `u2net` / `u2netp`).
3. Optionally set `MODEL` (`isnet-general-use`, `isnet-anime`, `u2net`, or `u2netp`) in Railway's **Variables** tab; Railway passes it as a Docker `ARG` at build time. Changing `MODEL` requires triggering a rebuild.
4. If exposing the service publicly, place TLS, authentication, and IP rate limiting at your ingress proxy.

## Upstream References & Licensing

- Source code in this repository is licensed under the [MIT License](LICENSE).
- [onnxruntime_go v1.22.0](https://github.com/yalue/onnxruntime_go/tree/v1.22.0) and [Microsoft ONNX Runtime v1.22.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0) (MIT; notices retained in `/app/licenses/`).
- [DIS / IS-Net](https://github.com/xuebinqin/DIS), [U-2-Net](https://github.com/xuebinqin/U-2-Net), and [rembg model weights](https://github.com/danielgatis/rembg) (Apache-2.0 / upstream dataset terms apply when redistributing weights).
