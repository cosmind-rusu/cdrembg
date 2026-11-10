// Package processor owns the segmentation model session and image pipeline.
package processor

import (
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"math"
	"sync"
)

// MaxPixels caps decoded dimensions at 16 megapixels (e.g. 4608x3456), covering
// standard 12 MP and 16 MP mobile photographs while bounding RGBA memory to 64 MiB.
const MaxPixels = 16_000_000

var (
	ErrImage              = errors.New("invalid image")
	ErrUnsupportedFormat  = fmt.Errorf("%w: upload a valid JPEG or PNG image", ErrImage)
	ErrDimensionsExceeded = fmt.Errorf("%w: image exceeds the 16-megapixel limit", ErrImage)
	ErrCorruptImage       = fmt.Errorf("%w: image data is truncated or corrupt", ErrImage)
)

// Config configures the native ONNX Runtime session and model profile.
type Config struct {
	LibraryPath  string
	ModelDir     string
	ModelPath    string
	CacheDir     string
	ModelName    string
	Threads      int
	AutoDownload bool
}

type sessionBackend interface {
	inputData() []float32
	outputData() []float32
	run() error
	close() error
}

type span1D struct {
	start  int
	offset int
	count  int
}

type sampleCoord struct {
	i0, i1 int
	f      float32
}

type scratchBuffers struct {
	xSpans   []span1D
	ySpans   []span1D
	xWeights []float32
	yWeights []float32
	rowAcc   []float32
	maskX    []sampleCoord
}

type pngBufferPool struct {
	pool sync.Pool
}

func (p *pngBufferPool) Get() *png.EncoderBuffer {
	if v := p.pool.Get(); v != nil {
		return v.(*png.EncoderBuffer)
	}
	return new(png.EncoderBuffer)
}

func (p *pngBufferPool) Put(b *png.EncoderBuffer) {
	p.pool.Put(b)
}

// Processor serializes access to reusable native tensors and resampling buffers.
// Close must be called only after active Process calls have returned.
type Processor struct {
	mu      sync.Mutex
	backend sessionBackend
	profile modelProfile
	scratch scratchBuffers
	pngPool pngBufferPool
}

// New initializes ONNX Runtime and binds preallocated input/output tensors for
// the configured model profile.
func New(cfg Config) (*Processor, error) {
	if cfg.ModelName == "" {
		cfg.ModelName = "isnet-anime"
	}
	if cfg.Threads < 1 {
		cfg.Threads = 1
	}
	profile, err := profileFor(cfg.ModelName)
	if err != nil {
		return nil, err
	}
	resolvedPath, err := resolveModelPath(cfg, profile, "")
	if err != nil {
		return nil, err
	}
	cfg.ModelPath = resolvedPath

	backend, err := newBackend(cfg, profile.size)
	if err != nil {
		return nil, err
	}
	return newWithBackend(profile, backend), nil
}

func newWithBackend(profile modelProfile, backend sessionBackend) *Processor {
	return &Processor{
		backend: backend,
		profile: profile,
		scratch: scratchBuffers{
			xSpans: make([]span1D, profile.size),
			ySpans: make([]span1D, profile.size),
			rowAcc: make([]float32, 3*profile.size),
		},
	}
}

// Close releases native ONNX session and tensor allocations.
func (p *Processor) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.backend == nil {
		return nil
	}
	err := p.backend.close()
	p.backend = nil
	return err
}

// Process validates image headers before full pixel allocation, normalizes EXIF
// orientation, executes segmentation inference, applies the predicted alpha mask
// in place, and encodes a transparent PNG.
func (p *Processor) Process(ctx context.Context, src io.ReadSeeker, dst io.Writer) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.backend == nil {
		return errors.New("processor closed")
	}

	rawImg, orientation, err := decode(src)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	img := toUprightNRGBA(rawImg, orientation)
	rawImg = nil // allow GC to reclaim source YCbCr/RGBA buffers before PNG encoding

	preprocessInto(img, p.backend.inputData(), p.profile, &p.scratch)
	if err := p.backend.run(); err != nil {
		return fmt.Errorf("inference: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	mask := p.backend.outputData()
	if err := normalizeMask(mask); err != nil {
		return err
	}
	applyMaskInPlace(img, mask, p.profile.size, &p.scratch)
	if err := ctx.Err(); err != nil {
		return err
	}

	enc := png.Encoder{
		CompressionLevel: png.BestSpeed,
		BufferPool:       &p.pngPool,
	}
	return enc.Encode(dst, img)
}

func decode(src io.ReadSeeker) (image.Image, int, error) {
	if _, err := src.Seek(0, io.SeekStart); err != nil {
		return nil, 1, err
	}
	cfg, format, err := image.DecodeConfig(src)
	if err != nil || (format != "jpeg" && format != "png") {
		return nil, 1, ErrUnsupportedFormat
	}
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, 1, ErrCorruptImage
	}
	if int64(cfg.Width)*int64(cfg.Height) > MaxPixels {
		return nil, 1, ErrDimensionsExceeded
	}

	orientation := 1
	if format == "jpeg" {
		orientation = readJPEGOrientation(src)
	}
	if _, err := src.Seek(0, io.SeekStart); err != nil {
		return nil, 1, err
	}
	img, _, err := image.Decode(src)
	if err != nil {
		return nil, 1, ErrCorruptImage
	}
	return img, orientation, nil
}

// preprocessInto resamples unassociated RGB into planar float32 NCHW [1,3,S,S],
// divides by the image maximum, and applies model-specific channel statistics.
// Downsampling uses box/area integration to prevent edge aliasing; upsampling
// uses half-pixel-centered bilinear interpolation.
func preprocessInto(src *image.NRGBA, data []float32, profile modelProfile, sc *scratchBuffers) {
	size := profile.size
	srcW, srcH := src.Rect.Dx(), src.Rect.Dy()

	sc.xSpans, sc.xWeights = buildKernel1D(size, srcW, sc.xSpans, sc.xWeights[:0])
	sc.ySpans, sc.yWeights = buildKernel1D(size, srcH, sc.ySpans, sc.yWeights[:0])
	if cap(sc.rowAcc) < 3*size {
		sc.rowAcc = make([]float32, 3*size)
	}
	acc := sc.rowAcc[:3*size]

	plane := size * size
	rPlane := data[:plane]
	gPlane := data[plane : 2*plane]
	bPlane := data[2*plane : 3*plane]

	var maximum float32
	for y := range size {
		clear(acc)
		ys := sc.ySpans[y]
		yw := sc.yWeights[ys.offset : ys.offset+ys.count]

		for ky, wy := range yw {
			sRow := src.Pix[(ys.start+ky)*src.Stride:]
			for x := range size {
				xs := sc.xSpans[x]
				xw := sc.xWeights[xs.offset : xs.offset+xs.count]
				var sr, sg, sb float32
				for kx, wx := range xw {
					pi := (xs.start + kx) * 4
					sr += float32(sRow[pi]) * wx
					sg += float32(sRow[pi+1]) * wx
					sb += float32(sRow[pi+2]) * wx
				}
				ai := x * 3
				acc[ai] += sr * wy
				acc[ai+1] += sg * wy
				acc[ai+2] += sb * wy
			}
		}

		rowOff := y * size
		for x := range size {
			ai := x * 3
			rv, gv, bv := acc[ai], acc[ai+1], acc[ai+2]
			rPlane[rowOff+x] = rv
			gPlane[rowOff+x] = gv
			bPlane[rowOff+x] = bv
			maximum = max(maximum, rv, gv, bv)
		}
	}

	if maximum == 0 {
		maximum = 1
	}
	invMax := 1 / maximum
	for ch := range 3 {
		mean := profile.means[ch]
		invStd := 1 / profile.stds[ch]
		chSlice := data[ch*plane : (ch+1)*plane]
		for i, v := range chSlice {
			chSlice[i] = (v*invMax - mean) * invStd
		}
	}
}

func buildKernel1D(dstSize, srcSize int, spans []span1D, weights []float32) ([]span1D, []float32) {
	if cap(spans) < dstSize {
		spans = make([]span1D, dstSize)
	} else {
		spans = spans[:dstSize]
	}

	if srcSize <= dstSize {
		for i := range dstSize {
			i0, i1, f := coordinate(i, dstSize, srcSize)
			off := len(weights)
			if i0 == i1 || f == 0 {
				weights = append(weights, 1)
				spans[i] = span1D{start: i0, offset: off, count: 1}
			} else {
				weights = append(weights, 1-f, f)
				spans[i] = span1D{start: i0, offset: off, count: 2}
			}
		}
		return spans, weights
	}

	scale := float64(srcSize) / float64(dstSize)
	invScale := 1.0 / scale
	for i := range dstSize {
		start := float64(i) * scale
		end := float64(i+1) * scale
		k0 := int(start)
		k1 := min(srcSize-1, int(math.Ceil(end))-1)
		off := len(weights)
		for k := k0; k <= k1; k++ {
			left := max(start, float64(k))
			right := min(end, float64(k+1))
			weights = append(weights, float32((right-left)*invScale))
		}
		spans[i] = span1D{start: k0, offset: off, count: k1 - k0 + 1}
	}
	return spans, weights
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
	if hi-lo > 1e-6 {
		invRange := 1 / (hi - lo)
		for i, v := range mask {
			mask[i] = max(0, min(1, (v-lo)*invRange))
		}
		return nil
	}
	// Constant predictions retain their probability instead of dividing by zero.
	for i, v := range mask {
		mask[i] = max(0, min(1, v))
	}
	return nil
}

// applyMaskInPlace bilinearly expands the [size x size] alpha mask to dst's
// dimensions and multiplies existing pixel alpha in place without allocating
// a second image buffer.
func applyMaskInPlace(dst *image.NRGBA, mask []float32, size int, sc *scratchBuffers) {
	w, h := dst.Rect.Dx(), dst.Rect.Dy()
	if cap(sc.maskX) < w {
		sc.maskX = make([]sampleCoord, w)
	}
	xs := sc.maskX[:w]
	for x := range w {
		i0, i1, f := coordinate(x, w, size)
		xs[x] = sampleCoord{i0: i0, i1: i1, f: f}
	}

	for y := range h {
		y0, y1, fy := coordinate(y, h, size)
		row0 := mask[y0*size : (y0+1)*size]
		row1 := mask[y1*size : (y1+1)*size]
		pixRow := dst.Pix[y*dst.Stride : y*dst.Stride+w*4]

		for x, s := range xs {
			top := row0[s.i0] + (row0[s.i1]-row0[s.i0])*s.f
			bot := row1[s.i0] + (row1[s.i1]-row1[s.i0])*s.f
			alpha := top + (bot-top)*fy
			ai := x*4 + 3
			if a := pixRow[ai]; a != 0 {
				pixRow[ai] = uint8(float32(a)*alpha + 0.5)
			}
		}
	}
}

func coordinate(position, target, source int) (int, int, float32) {
	v := max(0, min(float64(source-1), (float64(position)+0.5)*float64(source)/float64(target)-0.5))
	lo := int(v)
	return lo, min(lo+1, source-1), float32(v - float64(lo))
}
