// Package processor owns the native segmentation session and image transformation.
package processor

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const Size = 320
const MaxPixels = 4_000_000

var ErrImage = errors.New("use a valid JPEG or PNG image of at most 4 megapixels")

// Processor serializes access to reusable tensors. Close must follow HTTP shutdown.
type Processor struct {
	mu            sync.Mutex
	session       *ort.AdvancedSession
	input, output *ort.Tensor[float32]
	profile       modelProfile
}

type modelProfile struct {
	size  int
	isnet bool
	anime bool
}

func profileFor(name string) (modelProfile, error) {
	switch name {
	case "u2net", "u2netp":
		return modelProfile{size: Size}, nil
	case "isnet-general-use":
		return modelProfile{size: 1024, isnet: true}, nil
	case "isnet-anime":
		return modelProfile{size: 1024, anime: true}, nil
	default:
		return modelProfile{}, fmt.Errorf("unsupported MODEL_NAME %q (use u2net, u2netp, isnet-general-use, or isnet-anime)", name)
	}
}

func New(path string, threads int) (_ *Processor, err error) {
	return NewWithModel(path, threads, "isnet-general-use")
}

// NewWithModel binds the model's preprocessing and output semantics explicitly.
// IS-Net and U2-Net use different resolutions and channel normalization.
func NewWithModel(path string, threads int, name string) (_ *Processor, err error) {
	profile, err := profileFor(name)
	if err != nil {
		return nil, err
	}
	size := int64(profile.size)
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	for _, configure := range []func() error{
		func() error { return opts.SetIntraOpNumThreads(threads) },
		func() error { return opts.SetInterOpNumThreads(1) },
		func() error { return opts.SetExecutionMode(ort.ExecutionModeSequential) },
		func() error { return opts.SetCpuMemArena(false) },
		func() error { return opts.SetMemPattern(false) },
		func() error { return opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableBasic) },
	} {
		if err := configure(); err != nil {
			return nil, err
		}
	}
	// The wrapper destroys the temporary metadata session before we load the model.
	inputs, outputs, err := ort.GetInputOutputInfoWithOptions(path, opts)
	if err != nil {
		return nil, fmt.Errorf("inspect model: %w", err)
	}
	if len(inputs) != 1 || len(outputs) == 0 || !compatible(inputs[0], []int64{1, 3, size, size}) || !compatible(outputs[0], []int64{1, 1, size, size}) {
		return nil, fmt.Errorf("%s requires float32 NCHW input [1,3,%d,%d] and output [1,1,%d,%d]", name, size, size, size, size)
	}
	p := &Processor{profile: profile}
	defer func() {
		if err != nil {
			_ = p.Close()
		}
	}()
	p.input, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 3, size, size))
	if err != nil {
		return nil, err
	}
	p.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1, size, size))
	if err != nil {
		return nil, err
	}
	p.session, err = ort.NewAdvancedSession(path, []string{inputs[0].Name}, []string{outputs[0].Name}, []ort.Value{p.input}, []ort.Value{p.output}, opts)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return p, nil
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

func (p *Processor) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	var errs []error
	if p.session != nil {
		errs = append(errs, p.session.Destroy())
		p.session = nil
	}
	if p.output != nil {
		errs = append(errs, p.output.Destroy())
		p.output = nil
	}
	if p.input != nil {
		errs = append(errs, p.input.Destroy())
		p.input = nil
	}
	return errors.Join(errs...)
}

// Process checks dimensions before allocating decoded pixels, then writes a PNG.
// Native Run is synchronous and cannot be interrupted by this wrapper version.
func (p *Processor) Process(ctx context.Context, src io.ReadSeeker, dst io.Writer) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.session == nil {
		return errors.New("processor closed")
	}
	img, err := decode(src)
	if err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if p.profile.anime {
		preprocessNormalized(img, p.input.GetData(), p.profile.size, [3]float32{.485, .456, .406}, [3]float32{1, 1, 1})
	} else if p.profile.isnet {
		preprocessNormalized(img, p.input.GetData(), p.profile.size, [3]float32{.5, .5, .5}, [3]float32{1, 1, 1})
	} else {
		preprocessSize(img, p.input.GetData(), p.profile.size)
	}
	if err = p.session.Run(); err != nil {
		return fmt.Errorf("inference: %w", err)
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	mask := p.output.GetData()
	if err = normalizeMask(mask); err != nil {
		return err
	}
	result := applyMaskSize(img, mask, p.profile.size)
	if err = ctx.Err(); err != nil {
		return err
	}
	return (&png.Encoder{CompressionLevel: png.BestSpeed}).Encode(dst, result)
}

func decode(src io.ReadSeeker) (image.Image, error) {
	cfg, format, err := image.DecodeConfig(src)
	if err != nil || (format != "jpeg" && format != "png") || cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > MaxPixels {
		return nil, ErrImage
	}
	if _, err = src.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	img, _, err := image.Decode(src)
	if err != nil {
		return nil, ErrImage
	}
	return img, nil
}

// U2-Net expects planar RGB divided by the image maximum, then ImageNet normalization.
// Bilinear sampling uses unassociated RGB; existing alpha is preserved separately.
func preprocess(img image.Image, data []float32) {
	preprocessSize(img, data, Size)
}

func preprocessSize(img image.Image, data []float32, size int) {
	preprocessNormalized(img, data, size, [3]float32{.485, .456, .406}, [3]float32{.229, .224, .225})
}

func preprocessNormalized(img image.Image, data []float32, size int, means, stds [3]float32) {
	bounds := img.Bounds()
	var maximum float32
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			x0, x1, fx := coordinate(x, size, bounds.Dx())
			y0, y1, fy := coordinate(y, size, bounds.Dy())
			a := color.NRGBAModel.Convert(img.At(bounds.Min.X+x0, bounds.Min.Y+y0)).(color.NRGBA)
			b := color.NRGBAModel.Convert(img.At(bounds.Min.X+x1, bounds.Min.Y+y0)).(color.NRGBA)
			c := color.NRGBAModel.Convert(img.At(bounds.Min.X+x0, bounds.Min.Y+y1)).(color.NRGBA)
			d := color.NRGBAModel.Convert(img.At(bounds.Min.X+x1, bounds.Min.Y+y1)).(color.NRGBA)
			for ch, v := range [3]float32{blend(float32(a.R), float32(b.R), float32(c.R), float32(d.R), fx, fy), blend(float32(a.G), float32(b.G), float32(c.G), float32(d.G), fx, fy), blend(float32(a.B), float32(b.B), float32(c.B), float32(d.B), fx, fy)} {
				data[ch*size*size+y*size+x] = v
				maximum = max(maximum, v)
			}
		}
	}
	if maximum == 0 {
		maximum = 1
	}
	for ch := 0; ch < 3; ch++ {
		for i := 0; i < size*size; i++ {
			j := ch*size*size + i
			data[j] = (data[j]/maximum - means[ch]) / stds[ch]
		}
	}
}

func normalizeMask(mask []float32) error {
	lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
	for _, v := range mask {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("model returned a non-finite mask")
		}
		lo = min(lo, v)
		hi = max(hi, v)
	}
	for i, v := range mask {
		if hi-lo > 1e-6 {
			v = (v - lo) / (hi - lo)
		}
		// Constant predictions retain their probability instead of dividing by zero.
		mask[i] = max(0, min(1, v))
	}
	return nil
}

func applyMask(img image.Image, mask []float32) *image.NRGBA {
	return applyMaskSize(img, mask, Size)
}

func applyMaskSize(img image.Image, mask []float32, size int) *image.NRGBA {
	b := img.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		y0, y1, fy := coordinate(y, b.Dy(), size)
		for x := 0; x < b.Dx(); x++ {
			x0, x1, fx := coordinate(x, b.Dx(), size)
			alpha := blend(mask[y0*size+x0], mask[y0*size+x1], mask[y1*size+x0], mask[y1*size+x1], fx, fy)
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			c.A = uint8(math.Round(float64(float32(c.A) * alpha)))
			dst.SetNRGBA(x, y, c)
		}
	}
	return dst
}

func coordinate(position, target, source int) (int, int, float32) {
	v := max(0, min(float64(source-1), (float64(position)+.5)*float64(source)/float64(target)-.5))
	lo := int(v)
	return lo, min(lo+1, source-1), float32(v - float64(lo))
}
func blend(a, b, c, d, x, y float32) float32 { return (a+(b-a)*x)*(1-y) + (c+(d-c)*x)*y }
