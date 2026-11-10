package processor

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	defaultModelBaseURL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"
	maxModelBytes       = 256 << 20
)

type modelProfile struct {
	name     string
	filename string
	sha256   string
	size     int
	means    [3]float32
	stds     [3]float32
}

func profileFor(name string) (modelProfile, error) {
	switch name {
	case "isnet-anime":
		return modelProfile{
			name:     name,
			filename: "isnet-anime.onnx",
			sha256:   "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
			size:     1024,
			means:    [3]float32{0.485, 0.456, 0.406},
			stds:     [3]float32{1, 1, 1},
		}, nil
	case "isnet-general-use":
		return modelProfile{
			name:     name,
			filename: "isnet-general-use.onnx",
			sha256:   "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
			size:     1024,
			means:    [3]float32{0.5, 0.5, 0.5},
			stds:     [3]float32{1, 1, 1},
		}, nil
	case "u2net":
		return modelProfile{
			name:     name,
			filename: "u2net.onnx",
			sha256:   "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
			size:     320,
			means:    [3]float32{0.485, 0.456, 0.406},
			stds:     [3]float32{0.229, 0.224, 0.225},
		}, nil
	case "u2netp":
		return modelProfile{
			name:     name,
			filename: "u2netp.onnx",
			sha256:   "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
			size:     320,
			means:    [3]float32{0.485, 0.456, 0.406},
			stds:     [3]float32{0.229, 0.224, 0.225},
		}, nil
	default:
		return modelProfile{}, fmt.Errorf("unsupported MODEL %q (use isnet-anime, isnet-general-use, u2net, or u2netp)", name)
	}
}

func resolveModelPath(cfg Config, profile modelProfile, baseURL string) (string, error) {
	if cfg.ModelPath != "" {
		if _, err := os.Stat(cfg.ModelPath); err != nil {
			return "", fmt.Errorf("stat MODEL_PATH: %w", err)
		}
		return cfg.ModelPath, nil
	}

	modelDir := cfg.ModelDir
	if modelDir == "" {
		modelDir = "models"
	}
	baked := filepath.Join(modelDir, profile.filename)
	if info, err := os.Stat(baked); err == nil && !info.IsDir() {
		return baked, nil
	}

	if !cfg.AutoDownload {
		return "", fmt.Errorf("model file %q not found", baked)
	}

	cacheDir := cfg.CacheDir
	if cacheDir == "" {
		cacheDir = filepath.Join(os.TempDir(), "cdrembg-models")
	}
	cached := filepath.Join(cacheDir, profile.filename)
	if fileMatchesSHA256(cached, profile.sha256) {
		return cached, nil
	}

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", fmt.Errorf("prepare model cache (build with --build-arg MODEL=%s for read-only containers): %w", profile.name, err)
	}

	if baseURL == "" {
		baseURL = defaultModelBaseURL
	}
	url := baseURL + "/" + profile.filename
	slog.Info("downloading model weights", "model", profile.name, "url", url)

	if err := downloadVerified(url, cached, profile.sha256); err != nil {
		return "", fmt.Errorf("download model %s: %w", profile.name, err)
	}
	return cached, nil
}

func fileMatchesSHA256(path, expectedHex string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false
	}
	return hex.EncodeToString(h.Sum(nil)) == expectedHex
}

func downloadVerified(url, dstPath, expectedHex string) error {
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected HTTP status %d", resp.StatusCode)
	}

	tmp, err := os.CreateTemp(filepath.Dir(dstPath), ".model-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := true
	defer func() {
		_ = tmp.Close()
		if cleanup {
			_ = os.Remove(tmpPath)
		}
	}()

	h := sha256.New()
	written, err := io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, maxModelBytes+1))
	if err != nil {
		return err
	}
	if written > maxModelBytes {
		return errors.New("downloaded model exceeds size limit")
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != expectedHex {
		return fmt.Errorf("SHA-256 mismatch: got %s, expected %s", got, expectedHex)
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, dstPath); err != nil {
		return err
	}
	cleanup = false
	return nil
}
