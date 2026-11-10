# cdrembg

[![CI](https://github.com/cosmind-rusu/cdrembg/actions/workflows/ci.yml/badge.svg)](https://github.com/cosmind-rusu/cdrembg/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-17603e.svg)](LICENSE)

Servicio autoalojado en Go y ONNX Runtime para eliminar el fondo de imágenes desde el navegador o por API REST. Todo el procesado ocurre en memoria, sin guardar archivos en disco y devolviendo un PNG transparente a resolución original.

## Ejecución con Docker

```sh
docker build -t cdrembg .
docker run --rm -p 8080:8080 cdrembg
```

Abre `http://localhost:8080` y selecciona, arrastra o pega (`Ctrl+V`) cualquier imagen JPEG o PNG.

## Selección de modelo

No necesitas configurar nada más. Por defecto viene preinstalado **`isnet-anime`** (el modelo recomendado por la limpieza de sus recortes y contornos). Si quieres usar otro, basta con indicar la variable `MODEL`:

| `MODEL` | Resolución | Uso recomendado |
|---|---|---|
| `isnet-anime` *(por defecto)* | `1024×1024` | Recomendado. Contornos muy limpios en ilustraciones, personajes y recortes generales |
| `isnet-general-use` | `1024×1024` | Alternativa enfocada en fotografía general y producto |
| `u2net` | `320×320` | Inferencia más rápida en CPU con menor detalle fino |
| `u2netp` | `320×320` | Versión ultraligera (~4.6 MB) para servidores con poca memoria |

Puedes cambiar de modelo directamente al arrancar el contenedor (se descargará y verificará por SHA-256 si no estaba incluido en la imagen):

```sh
docker run --rm -p 8080:8080 -e MODEL=isnet-general-use cdrembg
```

O dejarlo integrado desde el build para que el contenedor arranque sin necesitar red:

```sh
docker build --build-arg MODEL=isnet-general-use -t cdrembg .
```

### Despliegue en Railway

Conecta el repositorio de GitHub en Railway (`railway.json` configura el `Dockerfile` y el healthcheck automáticamente). Si quieres cambiar el modelo, añade la variable `MODEL` en la pestaña **Variables** del servicio; para los modelos `1024×1024` se recomienda asignar entre 2 y 3 GB de RAM.

## API REST

```sh
curl -f -F 'file=@foto.jpg' http://localhost:8080/api/remove -o sin-fondo.png
```

- `GET /`: interfaz web integrada.
- `GET /health`: comprobación de estado (`200 OK`, `{"status":"ok"}`).
- `POST /api/remove`: campo multipart `file` (JPEG o PNG de hasta 10 MiB y 16 MP). Corrige automáticamente la orientación EXIF, conserva transparencias previas y responde `image/png` (o JSON con código `400`, `413`, `422` o `503` con cabecera `Retry-After` si hay otra inferencia en curso).

## Detalles técnicos

- **Binario único sin dependencias externas en cliente:** interfaz HTML/CSS/JS embebida con `embed.FS` y cabecera `Content-Security-Policy` estricta (`default-src 'none'`).
- **Pipeline de imagen con `0 allocs/op`:** acceso directo a buffers `*image.NRGBA`, escalado con integración de área (anti-aliasing), composición alfa *in-place* y `png.EncoderBufferPool`.
- **Memoria acotada y *load shedding*:** validación de cabeceras (`image.DecodeConfig`) antes de reservar píxeles, separación entre lectura de red e inferencia, y rechazo inmediato `503` en vez de acumular colas en RAM.
- **Contenedor endurecido:** verificación SHA-256 de ONNX Runtime `1.22.0` (`amd64`/`arm64`) y de los modelos, ejecución sin privilegios (`65532:65532`) y healthcheck integrado en el propio binario.
- **Tests portables sin CGO:** bindings nativos aislados mediante *build tags* para ejecutar tests y benchmarks con `CGO_ENABLED=0 go test ./...`.

## Licencia

Código bajo licencia [MIT](LICENSE). Incluye avisos de [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) y utiliza los pesos publicados por [DIS / IS-Net](https://github.com/xuebinqin/DIS), [U-2-Net](https://github.com/xuebinqin/U-2-Net) y [rembg](https://github.com/danielgatis/rembg).
