package processor

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"testing"
)

type mockBackend struct {
	in     []float32
	out    []float32
	fill   float32
	closed bool
}

func newMockBackend(size int, fill float32) *mockBackend {
	return &mockBackend{
		in:   make([]float32, 3*size*size),
		out:  make([]float32, size*size),
		fill: fill,
	}
}

func (m *mockBackend) inputData() []float32  { return m.in }
func (m *mockBackend) outputData() []float32 { return m.out }
func (m *mockBackend) run() error {
	for i := range m.out {
		m.out[i] = m.fill
	}
	return nil
}
func (m *mockBackend) close() error {
	m.closed = true
	return nil
}

func TestPreprocessBlackAndPlanarRGB(t *testing.T) {
	profile, err := profileFor("u2net")
	if err != nil {
		t.Fatal(err)
	}
	var sc scratchBuffers
	for _, c := range []color.NRGBA{{0, 0, 0, 255}, {255, 128, 64, 255}} {
		img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		img.SetNRGBA(0, 0, c)
		data := make([]float32, 3*profile.size*profile.size)
		preprocessInto(img, data, profile, &sc)

		maxVal := float32(max(c.R, c.G, c.B))
		if maxVal == 0 {
			maxVal = 1
		}
		plane := profile.size * profile.size
		for ch, value := range []uint8{c.R, c.G, c.B} {
			want := (float32(value)/maxVal - profile.means[ch]) / profile.stds[ch]
			for _, idx := range []int{ch * plane, (ch+1)*plane - 1} {
				if math.IsNaN(float64(data[idx])) || math.Abs(float64(data[idx]-want)) > 1e-5 {
					t.Fatalf("channel %d: got %v want %v", ch, data[idx], want)
				}
			}
		}
	}
}

func TestDownsampleAreaIntegration(t *testing.T) {
	// A 4x4 checkerboard of 0 and 200 downsampled to 1x1 must integrate all 16
	// pixels to their exact mean (100), avoiding point-sampling aliasing.
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := range 4 {
		for x := range 4 {
			v := uint8(0)
			if (x+y)%2 == 1 {
				v = 200
			}
			src.SetNRGBA(x, y, color.NRGBA{R: v, G: 100, B: 200, A: 255})
		}
	}

	profile := modelProfile{name: "test", size: 1, means: [3]float32{0, 0, 0}, stds: [3]float32{1, 1, 1}}
	var sc scratchBuffers
	data := make([]float32, 3)
	preprocessInto(src, data, profile, &sc)

	// Averaged channels are R=100, G=100, B=200 -> max=200 -> normalized [0.5, 0.5, 1.0].
	want := [3]float32{0.5, 0.5, 1.0}
	for ch := range 3 {
		if math.Abs(float64(data[ch]-want[ch])) > 1e-5 {
			t.Fatalf("channel %d: got %v want %v", ch, data[ch], want[ch])
		}
	}
}

func TestMaskPreservesColorAndExistingAlpha(t *testing.T) {
	raw := image.NewNRGBA(image.Rect(5, 7, 7, 8))
	raw.SetNRGBA(5, 7, color.NRGBA{200, 100, 50, 128})
	raw.SetNRGBA(6, 7, color.NRGBA{30, 60, 90, 0})

	img := toUprightNRGBA(raw, 1)
	const size = 320
	mask := make([]float32, size*size)
	for i := range mask {
		mask[i] = 0.5
	}
	var sc scratchBuffers
	applyMaskInPlace(img, mask, size, &sc)

	if img.Bounds() != image.Rect(0, 0, 2, 1) || img.NRGBAAt(0, 0) != (color.NRGBA{200, 100, 50, 64}) || img.NRGBAAt(1, 0).A != 0 {
		t.Fatalf("incorrect output: %v %v", img.Bounds(), img.Pix)
	}
}

func TestMaskNormalization(t *testing.T) {
	mask := []float32{2, 3, 4}
	if err := normalizeMask(mask); err != nil {
		t.Fatal(err)
	}
	if mask[0] != 0 || mask[1] != 0.5 || mask[2] != 1 {
		t.Fatal(mask)
	}
	constant := []float32{0.4, 0.4}
	if err := normalizeMask(constant); err != nil || constant[0] != 0.4 {
		t.Fatal(constant, err)
	}
	if normalizeMask([]float32{float32(math.NaN())}) == nil {
		t.Fatal("accepted NaN")
	}
	if normalizeMask([]float32{float32(math.Inf(1))}) == nil {
		t.Fatal("accepted Inf")
	}
}

func TestModelProfiles(t *testing.T) {
	var sc scratchBuffers

	isnet, err := profileFor("isnet-general-use")
	if err != nil || isnet.size != 1024 {
		t.Fatal(isnet, err)
	}
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{200, 100, 0, 255})
	data := make([]float32, 3)
	preprocessInto(img, data, modelProfile{size: 1, means: isnet.means, stds: isnet.stds}, &sc)
	if data[0] != 0.5 || data[1] != 0 || data[2] != -0.5 {
		t.Fatal(data)
	}

	anime, err := profileFor("isnet-anime")
	if err != nil || anime.size != 1024 {
		t.Fatal(anime, err)
	}
	img.SetNRGBA(0, 0, color.NRGBA{255, 0, 0, 255})
	preprocessInto(img, data, modelProfile{size: 1, means: anime.means, stds: anime.stds}, &sc)
	for i, want := range []float32{0.515, -0.456, -0.406} {
		if math.Abs(float64(data[i]-want)) > 1e-6 {
			t.Fatal(data)
		}
	}

	if _, err := profileFor("unknown-model"); err == nil {
		t.Fatal("expected error for unknown model")
	}
}

func TestDecodeValidation(t *testing.T) {
	if _, _, err := decode(bytes.NewReader([]byte("not an image"))); !errors.Is(err, ErrUnsupportedFormat) || !errors.Is(err, ErrImage) {
		t.Fatalf("unexpected format error: %v", err)
	}

	var b bytes.Buffer
	if err := png.Encode(&b, image.NewGray(image.Rect(0, 0, 4001, 4000))); err != nil {
		t.Fatal(err)
	}
	if _, _, err := decode(bytes.NewReader(b.Bytes())); !errors.Is(err, ErrDimensionsExceeded) || !errors.Is(err, ErrImage) {
		t.Fatalf("expected dimension limit error, got %v", err)
	}

	// Truncated PNG payload after valid IHDR header triggers ErrCorruptImage.
	b.Reset()
	_ = png.Encode(&b, image.NewNRGBA(image.Rect(0, 0, 64, 64)))
	truncated := b.Bytes()[:36]
	if _, _, err := decode(bytes.NewReader(truncated)); !errors.Is(err, ErrCorruptImage) || !errors.Is(err, ErrImage) {
		t.Fatalf("expected corrupt image error, got %v", err)
	}
}

func TestJPEGEXIFOrientation(t *testing.T) {
	// Create a 16x8 raw image whose bottom-left 8x4 quadrant is solid red (sized
	// above 2x2 so 4:2:0 JPEG chroma subsampling stays pure). Under EXIF
	// Orientation=6 (90° CW), raw bottom-left becomes visual top-left and
	// dimensions transpose from 16x8 to 8x16.
	raw := image.NewNRGBA(image.Rect(0, 0, 16, 8))
	for y := range 8 {
		for x := range 16 {
			raw.SetNRGBA(x, y, color.NRGBA{20, 40, 60, 255})
		}
	}
	for y := 4; y < 8; y++ {
		for x := range 8 {
			raw.SetNRGBA(x, y, color.NRGBA{250, 10, 10, 255})
		}
	}

	var rawJPEG bytes.Buffer
	if err := jpeg.Encode(&rawJPEG, raw, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatal(err)
	}
	withEXIF := injectEXIFOrientation(rawJPEG.Bytes(), 6)

	mock := newMockBackend(320, 1.0)
	profile, _ := profileFor("u2net")
	proc := newWithBackend(profile, mock)

	var out bytes.Buffer
	if err := proc.Process(context.Background(), bytes.NewReader(withEXIF), &out); err != nil {
		t.Fatal(err)
	}
	decoded, err := png.Decode(&out)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Bounds().Dx() != 8 || decoded.Bounds().Dy() != 16 {
		t.Fatalf("expected rotated 8x16 bounds, got %v", decoded.Bounds())
	}
	topLeft := color.NRGBAModel.Convert(decoded.At(1, 1)).(color.NRGBA)
	if topLeft.R < 230 || topLeft.B > 30 {
		t.Fatalf("expected red top-left quadrant after 90° CW EXIF rotation, got %+v", topLeft)
	}

	if err := proc.Close(); err != nil || !mock.closed {
		t.Fatalf("expected backend closed cleanly: err=%v closed=%v", err, mock.closed)
	}
}

func injectEXIFOrientation(jpegBytes []byte, orientation uint16) []byte {
	// Minimal TIFF IFD0 payload with 1 entry: Tag 0x0112 (Orientation), SHORT(3), count=1.
	payload := make([]byte, 6+8+2+12+4)
	copy(payload[:6], "Exif\x00\x00")
	tiff := payload[6:]
	copy(tiff[:4], "MM\x00\x2a")
	binary.BigEndian.PutUint32(tiff[4:8], 8)
	binary.BigEndian.PutUint16(tiff[8:10], 1)
	binary.BigEndian.PutUint16(tiff[10:12], 0x0112)
	binary.BigEndian.PutUint16(tiff[12:14], 3)
	binary.BigEndian.PutUint32(tiff[14:18], 1)
	binary.BigEndian.PutUint16(tiff[18:20], orientation)

	seg := make([]byte, 4+len(payload))
	seg[0], seg[1] = 0xff, 0xe1
	binary.BigEndian.PutUint16(seg[2:4], uint16(len(payload)+2))
	copy(seg[4:], payload)

	out := make([]byte, 0, len(jpegBytes)+len(seg))
	out = append(out, jpegBytes[:2]...)
	out = append(out, seg...)
	out = append(out, jpegBytes[2:]...)
	return out
}

func BenchmarkPreprocess4MP_ISNet(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 2000, 2000))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 180, 120, 60, 255
	}
	profile, _ := profileFor("isnet-general-use")
	data := make([]float32, 3*profile.size*profile.size)
	sc := scratchBuffers{
		xSpans: make([]span1D, profile.size),
		ySpans: make([]span1D, profile.size),
		rowAcc: make([]float32, 3*profile.size),
	}
	// Warm reusable weight slices once, matching steady-state Processor behavior.
	preprocessInto(img, data, profile, &sc)

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		preprocessInto(img, data, profile, &sc)
	}
}

func BenchmarkApplyMask4MP_ISNet(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 2000, 2000))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 180, 120, 60, 255
	}
	const size = 1024
	mask := make([]float32, size*size)
	for i := range mask {
		mask[i] = 0.75
	}
	var sc scratchBuffers
	applyMaskInPlace(img, mask, size, &sc)

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		applyMaskInPlace(img, mask, size, &sc)
	}
}
