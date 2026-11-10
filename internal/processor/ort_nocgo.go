//go:build !cgo

package processor

import "errors"

func newBackend(_ Config, _ int) (sessionBackend, error) {
	return nil, errors.New("ONNX Runtime inference requires building with CGO_ENABLED=1")
}
