package main

import (
	"bytes"
	"cdrembg/internal/processor"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

//go:embed static/index.html
var assets embed.FS

const maxUpload = 10 << 20

type imageProcessor interface {
	Process(context.Context, io.ReadSeeker, io.Writer) error
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	if err := run(); err != nil {
		slog.Error("server stopped", "error", err)
		os.Exit(1)
	}
}
func run() error {
	port := env("PORT", "8080")
	n, err := strconv.Atoi(port)
	if err != nil || n < 1 || n > 65535 {
		return errors.New("PORT must be between 1 and 65535")
	}
	threads, err := strconv.Atoi(env("ORT_THREADS", "1"))
	if err != nil || threads < 1 || threads > 8 {
		return errors.New("ORT_THREADS must be between 1 and 8")
	}
	ort.SetSharedLibraryPath(env("ONNX_RUNTIME_LIBRARY_PATH", "lib/libonnxruntime.so.1.22.0"))
	if err = ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize ONNX Runtime: %w", err)
	}
	defer ort.DestroyEnvironment()
	modelName := env("MODEL_NAME", "isnet-general-use")
	p, err := processor.NewWithModel(env("MODEL_PATH", "models/"+modelName+".onnx"), threads, modelName)
	if err != nil {
		return err
	}
	defer p.Close()
	server := &http.Server{Addr: ":" + port, Handler: newHandler(p), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 180 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 << 10}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	stopped := make(chan error, 1)
	go func() { stopped <- server.ListenAndServe() }()
	slog.Info("listening", "port", port, "model", modelName)
	select {
	case err = <-stopped:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		slog.Info("draining requests")
		// Do not destroy native tensors during inference. The orchestrator's grace
		// period supplies the hard deadline if native execution cannot finish.
		return server.Shutdown(context.Background())
	}
}

func newHandler(p imageProcessor) http.Handler {
	mux := http.NewServeMux()
	gate := make(chan struct{}, 1)
	page, _ := assets.ReadFile("static/index.html")
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(page)
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("POST /api/remove", func(w http.ResponseWriter, r *http.Request) {
		select {
		case gate <- struct{}{}:
			defer func() { <-gate }()
		default:
			w.Header().Set("Retry-After", "5")
			apiError(w, 503, "The processor is busy. Try again shortly.")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxUpload+(64<<10))
		defer r.Body.Close()
		// The entire bounded request fits in memory, avoiding image files on disk.
		err := r.ParseMultipartForm(maxUpload + (64 << 10))
		if r.MultipartForm != nil {
			defer r.MultipartForm.RemoveAll()
		}
		if err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				apiError(w, 413, "Upload exceeds 10 MiB.")
			} else {
				apiError(w, 400, "Expected multipart/form-data with one file field named file.")
			}
			return
		}
		files := r.MultipartForm.File["file"]
		count := 0
		for _, values := range r.MultipartForm.File {
			count += len(values)
		}
		if len(files) != 1 || count != 1 {
			apiError(w, 400, "Provide exactly one image in the file field.")
			return
		}
		if files[0].Size > maxUpload {
			apiError(w, 413, "Upload exceeds 10 MiB.")
			return
		}
		file, err := files[0].Open()
		if err != nil {
			apiError(w, 500, "Cannot read upload.")
			return
		}
		defer file.Close()
		var result bytes.Buffer
		started := time.Now()
		if err = p.Process(r.Context(), file, &result); err != nil {
			if errors.Is(err, processor.ErrImage) {
				apiError(w, 422, err.Error())
				return
			}
			if r.Context().Err() != nil {
				return
			}
			slog.Error("processing failed", "error", err)
			apiError(w, 500, "Image processing failed.")
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Disposition", `attachment; filename="background-removed.png"`)
		w.Header().Set("Content-Length", strconv.Itoa(result.Len()))
		_, _ = w.Write(result.Bytes())
		slog.Info("image processed", "duration_ms", time.Since(started).Milliseconds())
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Cache-Control", "no-store")
		mux.ServeHTTP(w, r)
	})
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
