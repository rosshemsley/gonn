package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"

	"gonum.org/v1/gonum/mat"
)

func LoadImagesGzipFile(path string) (*mat.Dense, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rgz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer rgz.Close()

	return LoadImages(rgz)
}

func LoadLabelsGzipFile(path string) (*mat.Dense, error) {
	r, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	rgz, err := gzip.NewReader(r)
	if err != nil {
		return nil, err
	}

	return LoadLabels(rgz)
}

// LoadImages reads images for mnist data.
// Returns a matrix where each row contains a new image.
// Images are unrolled into vector of length 28*28 from a row-major matrix.
func LoadImages(r io.Reader) (*mat.Dense, error) {
	var magic, n, rows, cols int32

	err := binary.Read(r, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}
	if magic != 2051 {
		return nil, fmt.Errorf("unexpected file format")
	}

	err = binary.Read(r, binary.BigEndian, &n)
	if err != nil {
		return nil, err
	}
	err = binary.Read(r, binary.BigEndian, &rows)
	if err != nil {
		return nil, err
	}
	err = binary.Read(r, binary.BigEndian, &cols)
	if err != nil {
		return nil, err
	}

	if rows != 28 || cols != 28 {
		return nil, fmt.Errorf("unpexected image size: %dX%d", rows, cols)
	}

	result := mat.NewDense(int(n), int(rows*cols), nil)
	for i := 0; i < int(n); i++ {
		img := make([]byte, rows*cols)
		imgVals := make([]float64, rows*cols)
		err := binary.Read(r, binary.BigEndian, &img)
		if err != nil {
			return nil, err
		}

		for i, v := range img {
			imgVals[i] = float64(v) / 255
		}

		result.SetRow(i, imgVals)
	}

	return result, nil
}

// LoadLabels returns a matrix with a row for each label loaded.
// The labels are encoded with one-hot encoding, and so each column has 10 entries.
func LoadLabels(r io.Reader) (*mat.Dense, error) {
	var magic, n int32

	err := binary.Read(r, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}
	if magic != 2049 {
		return nil, fmt.Errorf("unexpected file format")
	}

	err = binary.Read(r, binary.BigEndian, &n)
	if err != nil {
		return nil, err
	}

	result := mat.NewDense(int(n), 10, nil)
	for i := 0; i < int(n); i++ {
		var v int8
		err := binary.Read(r, binary.BigEndian, &v)
		if err != nil {
			return nil, err
		}
		if v < 0 || v > 9 {
			return nil, fmt.Errorf("invalid label: %d", v)
		}
		result.Set(i, int(v), 1)
	}

	return result, nil
}

func LabelValue(row []float64) int {
	maxIndex := -1
	max := -1.0
	for i, v := range row {
		if v > max {
			max = v
			maxIndex = i
		}
	}
	return maxIndex
}

func WritePNG(w io.Writer, vals []float64) {
	img := image.NewRGBA(image.Rect(0, 0, 28, 28))

	for i, v := range vals {
		iv := 255 - uint8(v)
		img.Set(i%28, i/28, color.RGBA{iv, iv, iv, 255})
	}
	png.Encode(w, img)
}
