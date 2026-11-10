//go:build cgo

package processor

import (
	"errors"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortEnvMu   sync.Mutex
	ortEnvRefs int
)

func acquireRuntime(libraryPath string) error {
	ortEnvMu.Lock()
	defer ortEnvMu.Unlock()
	if ortEnvRefs == 0 {
		if libraryPath != "" {
			ort.SetSharedLibraryPath(libraryPath)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("initialize ONNX Runtime: %w", err)
		}
	}
	ortEnvRefs++
	return nil
}

func releaseRuntime() error {
	ortEnvMu.Lock()
	defer ortEnvMu.Unlock()
	if ortEnvRefs <= 0 {
		return nil
	}
	ortEnvRefs--
	if ortEnvRefs == 0 {
		return ort.DestroyEnvironment()
	}
	return nil
}

type ortBackend struct {
	session       *ort.AdvancedSession
	input, output *ort.Tensor[float32]
}

func newBackend(cfg Config, size int) (_ sessionBackend, err error) {
	if err := acquireRuntime(cfg.LibraryPath); err != nil {
		return nil, err
	}
	b := &ortBackend{}
	defer func() {
		if err != nil {
			_ = b.close()
		}
	}()

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()

	for _, configure := range []func() error{
		func() error { return opts.SetIntraOpNumThreads(cfg.Threads) },
		func() error { return opts.SetInterOpNumThreads(1) },
		func() error { return opts.SetExecutionMode(ort.ExecutionModeSequential) },
		func() error { return opts.SetCpuMemArena(false) },
		func() error { return opts.SetMemPattern(false) },
		func() error { return opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll) },
	} {
		if err := configure(); err != nil {
			return nil, err
		}
	}

	dim := int64(size)
	inputs, outputs, err := ort.GetInputOutputInfoWithOptions(cfg.ModelPath, opts)
	if err != nil {
		return nil, fmt.Errorf("inspect model: %w", err)
	}
	if len(inputs) != 1 || len(outputs) == 0 || !compatible(inputs[0], []int64{1, 3, dim, dim}) || !compatible(outputs[0], []int64{1, 1, dim, dim}) {
		return nil, fmt.Errorf("%s requires float32 NCHW input [1,3,%d,%d] and output [1,1,%d,%d]", cfg.ModelName, dim, dim, dim, dim)
	}

	b.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, dim, dim))
	if err != nil {
		return nil, err
	}
	b.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1, dim, dim))
	if err != nil {
		return nil, err
	}
	b.session, err = ort.NewAdvancedSession(
		cfg.ModelPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		[]ort.Value{b.input},
		[]ort.Value{b.output},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return b, nil
}

func compatible(info ort.InputOutputInfo, shape []int64) bool {
	if info.OrtValueType != ort.ONNXTypeTensor || info.DataType != ort.TensorElementDataTypeFloat || len(info.Dimensions) != len(shape) {
		return false
	}
	for i, n := range info.Dimensions {
		if n > 0 && n != shape[i] {
			return false
		}
	}
	return true
}

func (b *ortBackend) inputData() []float32  { return b.input.GetData() }
func (b *ortBackend) outputData() []float32 { return b.output.GetData() }
func (b *ortBackend) run() error            { return b.session.Run() }

func (b *ortBackend) close() error {
	var errs []error
	if b.session != nil {
		errs = append(errs, b.session.Destroy())
		b.session = nil
	}
	if b.output != nil {
		errs = append(errs, b.output.Destroy())
		b.output = nil
	}
	if b.input != nil {
		errs = append(errs, b.input.Destroy())
		b.input = nil
	}
	errs = append(errs, releaseRuntime())
	return errors.Join(errs...)
}
