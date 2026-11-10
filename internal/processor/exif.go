package processor

import (
	"encoding/binary"
	"image"
	"image/color"
	"io"
)

// readJPEGOrientation extracts the EXIF Orientation tag (1..8) from a JPEG
// stream, returning 1 when no valid tag is present. The caller must seek the
// stream back to the start before full image decoding.
func readJPEGOrientation(r io.ReadSeeker) int {
	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return 1
	}
	var soi [2]byte
	if _, err := io.ReadFull(r, soi[:]); err != nil || soi[0] != 0xff || soi[1] != 0xd8 {
		return 1
	}

	var hdr [4]byte
	for range 32 {
		if _, err := io.ReadFull(r, hdr[:2]); err != nil || hdr[0] != 0xff {
			return 1
		}
		marker := hdr[1]
		for marker == 0xff {
			if _, err := io.ReadFull(r, hdr[1:2]); err != nil {
				return 1
			}
			marker = hdr[1]
		}
		if marker == 0xd9 || marker == 0xda {
			return 1
		}
		if _, err := io.ReadFull(r, hdr[2:4]); err != nil {
			return 1
		}
		segLen := int(binary.BigEndian.Uint16(hdr[2:4])) - 2
		if segLen < 0 {
			return 1
		}
		if marker == 0xe1 && segLen >= 14 && segLen <= 64<<10 {
			buf := make([]byte, segLen)
			if _, err := io.ReadFull(r, buf); err != nil {
				return 1
			}
			if orient := parseEXIFOrientation(buf); orient != 0 {
				return orient
			}
			continue
		}
		if _, err := r.Seek(int64(segLen), io.SeekCurrent); err != nil {
			return 1
		}
	}
	return 1
}

func parseEXIFOrientation(app1 []byte) int {
	if len(app1) < 14 || string(app1[:6]) != "Exif\x00\x00" {
		return 0
	}
	tiff := app1[6:]
	var order binary.ByteOrder
	switch string(tiff[:4]) {
	case "II\x2a\x00":
		order = binary.LittleEndian
	case "MM\x00\x2a":
		order = binary.BigEndian
	default:
		return 0
	}

	ifdOffset := int(order.Uint32(tiff[4:8]))
	if ifdOffset < 8 || ifdOffset+2 > len(tiff) {
		return 0
	}
	entries := int(order.Uint16(tiff[ifdOffset : ifdOffset+2]))
	pos := ifdOffset + 2
	for range entries {
		if pos+12 > len(tiff) {
			return 0
		}
		tag := order.Uint16(tiff[pos : pos+2])
		fieldType := order.Uint16(tiff[pos+2 : pos+4])
		count := order.Uint32(tiff[pos+4 : pos+8])
		if tag == 0x0112 && fieldType == 3 && count == 1 {
			v := int(order.Uint16(tiff[pos+8 : pos+10]))
			if v >= 1 && v <= 8 {
				return v
			}
			return 1
		}
		pos += 12
	}
	return 0
}

// toUprightNRGBA converts any decoded image to a zero-origin *image.NRGBA with
// EXIF orientation applied. Common formats (NRGBA, RGBA, YCbCr, Gray, Paletted)
// use direct slice loops to avoid per-pixel color.Color interface allocations.
func toUprightNRGBA(img image.Image, orientation int) *image.NRGBA {
	base := toNRGBA(img)
	if orientation <= 1 || orientation > 8 {
		return base
	}
	return applyOrientation(base, orientation)
}

func toNRGBA(img image.Image) *image.NRGBA {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))

	switch src := img.(type) {
	case *image.NRGBA:
		for y := range h {
			sOff := (b.Min.Y+y-src.Rect.Min.Y)*src.Stride + (b.Min.X-src.Rect.Min.X)*4
			dOff := y * dst.Stride
			copy(dst.Pix[dOff:dOff+w*4], src.Pix[sOff:sOff+w*4])
		}
	case *image.RGBA:
		for y := range h {
			sRow := src.Pix[(b.Min.Y+y-src.Rect.Min.Y)*src.Stride+(b.Min.X-src.Rect.Min.X)*4:]
			dRow := dst.Pix[y*dst.Stride:]
			for x := range w {
				si := x * 4
				r, g, bl, a := sRow[si], sRow[si+1], sRow[si+2], sRow[si+3]
				switch a {
				case 0xff:
					dRow[si], dRow[si+1], dRow[si+2], dRow[si+3] = r, g, bl, 0xff
				case 0:
					dRow[si], dRow[si+1], dRow[si+2], dRow[si+3] = 0, 0, 0, 0
				default:
					a32 := uint32(a)
					dRow[si] = uint8(((uint32(r)*0xffff)/a32 + 0x80) >> 8)
					dRow[si+1] = uint8(((uint32(g)*0xffff)/a32 + 0x80) >> 8)
					dRow[si+2] = uint8(((uint32(bl)*0xffff)/a32 + 0x80) >> 8)
					dRow[si+3] = a
				}
			}
		}
	case *image.YCbCr:
		for y := range h {
			sy := b.Min.Y + y
			dRow := dst.Pix[y*dst.Stride:]
			for x := range w {
				sx := b.Min.X + x
				yi := src.YOffset(sx, sy)
				ci := src.COffset(sx, sy)
				r, g, bl := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
				di := x * 4
				dRow[di], dRow[di+1], dRow[di+2], dRow[di+3] = r, g, bl, 0xff
			}
		}
	case *image.Gray:
		for y := range h {
			sRow := src.Pix[(b.Min.Y+y-src.Rect.Min.Y)*src.Stride+(b.Min.X-src.Rect.Min.X):]
			dRow := dst.Pix[y*dst.Stride:]
			for x := range w {
				v := sRow[x]
				di := x * 4
				dRow[di], dRow[di+1], dRow[di+2], dRow[di+3] = v, v, v, 0xff
			}
		}
	case *image.Paletted:
		var lut [256]color.NRGBA
		for i, c := range src.Palette {
			if i >= len(lut) {
				break
			}
			lut[i] = color.NRGBAModel.Convert(c).(color.NRGBA)
		}
		for y := range h {
			sRow := src.Pix[(b.Min.Y+y-src.Rect.Min.Y)*src.Stride+(b.Min.X-src.Rect.Min.X):]
			dRow := dst.Pix[y*dst.Stride:]
			for x := range w {
				c := lut[sRow[x]]
				di := x * 4
				dRow[di], dRow[di+1], dRow[di+2], dRow[di+3] = c.R, c.G, c.B, c.A
			}
		}
	default:
		for y := range h {
			dRow := dst.Pix[y*dst.Stride:]
			for x := range w {
				c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				di := x * 4
				dRow[di], dRow[di+1], dRow[di+2], dRow[di+3] = c.R, c.G, c.B, c.A
			}
		}
	}
	return dst
}

func applyOrientation(src *image.NRGBA, orientation int) *image.NRGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	outW, outH := w, h
	if orientation >= 5 {
		outW, outH = h, w
	}
	dst := image.NewNRGBA(image.Rect(0, 0, outW, outH))

	for y := range outH {
		dRow := dst.Pix[y*dst.Stride:]
		for x := range outW {
			var sx, sy int
			switch orientation {
			case 2:
				sx, sy = w-1-x, y
			case 3:
				sx, sy = w-1-x, h-1-y
			case 4:
				sx, sy = x, h-1-y
			case 5:
				sx, sy = y, x
			case 6:
				sx, sy = y, h-1-x
			case 7:
				sx, sy = w-1-y, h-1-x
			case 8:
				sx, sy = w-1-y, x
			default:
				sx, sy = x, y
			}
			sOff := sy*src.Stride + sx*4
			dOff := x * 4
			copy(dRow[dOff:dOff+4], src.Pix[sOff:sOff+4])
		}
	}
	return dst
}
