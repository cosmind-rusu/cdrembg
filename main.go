package main

import (
	"bytes"
	"cdrembg/internal/processor"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"syscall"
	"time"
)

//go:embed static/index.html static/styles.css static/app.js
var assets embed.FS

const (
	maxUpload       = 10 << 20
	multipartMargin = 64 << 10
	maxRequestBytes = maxUpload + multipartMargin
	cspHeaderValue  = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

type imageProcessor interface {
	Process(context.Context, io.ReadSeeker, io.Writer) error
}

func main() {
	if len(os.Args) == 2 && (os.Args[1] == "-healthcheck" || os.Args[1] == "--healthcheck") {
		if err := checkHealth("http://127.0.0.1:" + env("PORT", "8080") + "/health"); err != nil {
			os.Exit(1)
		}
		return
	}

	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	if err := run(); err != nil {
		slog.Error("server stopped", "error", err)
		os.Exit(1)
	}
}

func run() error {
	port := env("PORT", "8080")
	if err := validatePort(port); err != nil {
		return err
	}

	defaultThreads := strconv.Itoa(min(4, max(1, runtime.NumCPU())))
	threads, err := strconv.Atoi(env("ORT_THREADS", defaultThreads))
	if err != nil || threads < 1 || threads > 16 {
		return errors.New("ORT_THREADS must be between 1 and 16")
	}

	modelName := env("MODEL", env("MODEL_NAME", "isnet-anime"))
	p, err := processor.New(processor.Config{
		LibraryPath:  env("ONNX_RUNTIME_LIBRARY_PATH", "lib/libonnxruntime.so.1.22.0"),
		ModelDir:     env("MODEL_DIR", "models"),
		ModelPath:    os.Getenv("MODEL_PATH"),
		ModelName:    modelName,
		Threads:      threads,
		AutoDownload: true,
	})
	if err != nil {
		return err
	}
	defer p.Close()

	server := &http.Server{
		Addr:              ":" + port,
		Handler:           newHandler(p),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      180 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    16 << 10,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	stopped := make(chan error, 1)
	go func() {
		stopped <- server.ListenAndServe()
	}()
	slog.Info("listening", "port", port, "model", modelName, "threads", threads)

	select {
	case err := <-stopped:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		slog.Info("draining requests")
		// Wait for any active inference call to return before defer p.Close()
		// releases native ONNX tensors.
		return server.Shutdown(context.Background())
	}
}

func checkHealth(url string) error {
	if err := validatePort(env("PORT", "8080")); err != nil {
		return err
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected health status: %d", resp.StatusCode)
	}
	return nil
}

func validatePort(port string) error {
	n, err := strconv.Atoi(port)
	if err != nil || n < 1 || n > 65535 {
		return errors.New("PORT must be between 1 and 65535")
	}
	return nil
}

func newHandler(p imageProcessor) http.Handler {
	mux := http.NewServeMux()
	uploadGate := make(chan struct{}, 2)
	inferenceGate := make(chan struct{}, 1)

	indexHTML, _ := assets.ReadFile("static/index.html")
	stylesCSS, _ := assets.ReadFile("static/styles.css")
	appJS, _ := assets.ReadFile("static/app.js")

	mux.HandleFunc("GET /{$}", serveAsset("text/html; charset=utf-8", indexHTML))
	mux.HandleFunc("GET /styles.css", serveAsset("text/css; charset=utf-8", stylesCSS))
	mux.HandleFunc("GET /app.js", serveAsset("text/javascript; charset=utf-8", appJS))
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /api/remove", func(w http.ResponseWriter, r *http.Request) {
		if r.ContentLength > maxRequestBytes {
			apiError(w, http.StatusRequestEntityTooLarge, "Upload exceeds 10 MiB.")
			return
		}
		mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil || mediaType != "multipart/form-data" || params["boundary"] == "" {
			apiError(w, http.StatusBadRequest, "Expected multipart/form-data with one file field named file.")
			return
		}

		// Bound concurrent body reads independently so slow network uploads cannot
		// hold the CPU inference slot idle or exhaust memory.
		select {
		case uploadGate <- struct{}{}:
		default:
			w.Header().Set("Retry-After", "5")
			apiError(w, http.StatusServiceUnavailable, "The processor is busy. Try again shortly.")
			return
		}
		uploadReleased := false
		releaseUpload := func() {
			if !uploadReleased {
				uploadReleased = true
				<-uploadGate
			}
		}
		defer releaseUpload()

		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
		defer r.Body.Close()

		err = r.ParseMultipartForm(maxRequestBytes)
		if r.MultipartForm != nil {
			defer r.MultipartForm.RemoveAll()
		}
		if err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				apiError(w, http.StatusRequestEntityTooLarge, "Upload exceeds 10 MiB.")
			} else {
				apiError(w, http.StatusBadRequest, "Expected multipart/form-data with one file field named file.")
			}
			return
		}

		files := r.MultipartForm.File["file"]
		totalFiles := 0
		for _, values := range r.MultipartForm.File {
			totalFiles += len(values)
		}
		if len(files) != 1 || totalFiles != 1 || files[0].Size <= 0 {
			apiError(w, http.StatusBadRequest, "Provide exactly one non-empty image in the file field.")
			return
		}
		if files[0].Size > maxUpload {
			apiError(w, http.StatusRequestEntityTooLarge, "Upload exceeds 10 MiB.")
			return
		}

		file, err := files[0].Open()
		if err != nil {
			apiError(w, http.StatusInternalServerError, "Cannot read upload.")
			return
		}
		defer file.Close()
		releaseUpload()

		select {
		case inferenceGate <- struct{}{}:
			defer func() { <-inferenceGate }()
		default:
			w.Header().Set("Retry-After", "5")
			apiError(w, http.StatusServiceUnavailable, "The processor is busy. Try again shortly.")
			return
		}

		var result bytes.Buffer
		started := time.Now()
		if err := p.Process(r.Context(), file, &result); err != nil {
			if errors.Is(err, processor.ErrImage) {
				apiError(w, http.StatusUnprocessableEntity, err.Error())
				return
			}
			if r.Context().Err() != nil {
				return
			}
			slog.Error("processing failed", "error", err)
			apiError(w, http.StatusInternalServerError, "Image processing failed.")
			return
		}

		elapsedMs := time.Since(started).Milliseconds()
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Disposition", `attachment; filename="background-removed.png"`)
		w.Header().Set("Content-Length", strconv.Itoa(result.Len()))
		w.Header().Set("X-Process-Time-Ms", strconv.FormatInt(elapsedMs, 10))
		_, _ = w.Write(result.Bytes())
		slog.Info("image processed",
			"duration_ms", elapsedMs,
			"input_bytes", files[0].Size,
			"output_bytes", result.Len(),
		)
	})

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Security-Policy", cspHeaderValue)
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Cache-Control", "no-store")
		mux.ServeHTTP(w, r)
	})
}

func serveAsset(contentType string, body []byte) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write(body)
	}
}

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func apiError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
