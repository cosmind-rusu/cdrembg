package processor

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

func TestPreprocessBlackAndPlanarRGB(t *testing.T) {
	for _, c := range []color.NRGBA{{0, 0, 0, 255}, {255, 128, 64, 255}} {
		img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		img.SetNRGBA(0, 0, c)
		data := make([]float32, 3*Size*Size)
		preprocess(img, data)
		for channel, value := range []uint8{c.R, c.G, c.B} {
			want := (float32(value)/255 - []float32{.485, .456, .406}[channel]) / []float32{.229, .224, .225}[channel]
			for _, i := range []int{channel * Size * Size, (channel+1)*Size*Size - 1} {
				if math.IsNaN(float64(data[i])) || math.Abs(float64(data[i]-want)) > 1e-5 {
					t.Fatalf("channel %d: got %v want %v", channel, data[i], want)
				}
			}
		}
	}
}

func TestMaskPreservesColorAndExistingAlpha(t *testing.T) {
	img := image.NewNRGBA(image.Rect(5, 7, 7, 8))
	img.SetNRGBA(5, 7, color.NRGBA{200, 100, 50, 128})
	img.SetNRGBA(6, 7, color.NRGBA{30, 60, 90, 0})
	mask := make([]float32, Size*Size)
	for i := range mask {
		mask[i] = .5
	}
	got := applyMask(img, mask)
	if got.Bounds() != image.Rect(0, 0, 2, 1) || got.NRGBAAt(0, 0) != (color.NRGBA{200, 100, 50, 64}) || got.NRGBAAt(1, 0).A != 0 {
		t.Fatalf("incorrect output: %v %v", got.Bounds(), got.Pix)
	}
}

func TestMaskNormalization(t *testing.T) {
	mask := []float32{2, 3, 4}
	if err := normalizeMask(mask); err != nil {
		t.Fatal(err)
	}
	if mask[0] != 0 || mask[1] != .5 || mask[2] != 1 {
		t.Fatal(mask)
	}
	constant := []float32{.4, .4}
	if err := normalizeMask(constant); err != nil || constant[0] != .4 {
		t.Fatal(constant, err)
	}
	if normalizeMask([]float32{float32(math.NaN())}) == nil {
		t.Fatal("accepted NaN")
	}
}

func TestVariableResolutionPipeline(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{255, 0, 0, 128})
	const size = 1024
	data := make([]float32, 3*size*size)
	preprocessSize(img, data, size)
	if math.Abs(float64(data[len(data)-1]-(-.406/.225))) > 1e-5 {
		t.Fatal("incorrect final channel")
	}
	mask := make([]float32, size*size)
	for i := range mask {
		mask[i] = .5
	}
	if got := applyMaskSize(img, mask, size).NRGBAAt(0, 0); got != (color.NRGBA{255, 0, 0, 64}) {
		t.Fatal(got)
	}
}

func TestISNetNormalization(t *testing.T) {
	profile, err := profileFor("isnet-general-use")
	if err != nil || profile.size != 1024 || !profile.isnet {
		t.Fatal(profile, err)
	}
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{200, 100, 0, 255})
	data := make([]float32, 3)
	preprocessNormalized(img, data, 1, [3]float32{.5, .5, .5}, [3]float32{1, 1, 1})
	if data[0] != .5 || data[1] != 0 || data[2] != -.5 {
		t.Fatal(data)
	}
}

func TestAnimeNormalization(t *testing.T) {
	profile, err := profileFor("isnet-anime")
	if err != nil || profile.size != 1024 || !profile.anime || profile.isnet {
		t.Fatal(profile, err)
	}
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{255, 0, 0, 255})
	data := make([]float32, 3)
	preprocessNormalized(img, data, 1, [3]float32{.485, .456, .406}, [3]float32{1, 1, 1})
	for i, want := range []float32{.515, -.456, -.406} {
		if math.Abs(float64(data[i]-want)) > 1e-6 {
			t.Fatal(data)
		}
	}
}

func TestDecodeValidation(t *testing.T) {
	if _, err := decode(bytes.NewReader([]byte("not an image"))); err != ErrImage {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewGray(image.Rect(0, 0, 2001, 2000))); err != nil {
		t.Fatal(err)
	}
	if _, err := decode(bytes.NewReader(b.Bytes())); err != ErrImage {
		t.Fatal("accepted oversized dimensions", err)
	}
	b.Reset()
	_ = png.Encode(&b, image.NewGray(image.Rect(0, 0, 2, 3)))
	img, err := decode(bytes.NewReader(b.Bytes()))
	if err != nil || img.Bounds().Dx() != 2 {
		t.Fatal(err)
	}
}
