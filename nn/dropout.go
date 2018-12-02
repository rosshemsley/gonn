package nn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type DropoutLayer struct {
	mask         *mat.Dense
	p            float64
	trainingMode bool
}

func NewDropoutLayer(p float64) *DropoutLayer {
	return &DropoutLayer{p: p, trainingMode: true}
}

func (l *DropoutLayer) SetTrainingEnabled(b bool) {
	l.trainingMode = b
}

func (l *DropoutLayer) Forwards(x *mat.Dense) *mat.Dense {
	if !l.trainingMode {
		result := mat.DenseCopyOf(x)
		result.Scale(1-l.p, x)
		return result
	}

	rows, cols := x.Dims()
	l.mask = newRandomMask(cols, l.p)
	// log.Printf("new mask: %v", l.mask)

	result := mat.DenseCopyOf(x)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, x.At(i, j)*l.mask.At(0, j))
		}
	}

	return result
}

func (l *DropoutLayer) Backwards(grad *mat.Dense) *mat.Dense {
	rows, cols := grad.Dims()
	result := mat.DenseCopyOf(grad)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, result.At(i, j)*l.mask.At(0, j))
		}
	}

	return result
}

func (l *DropoutLayer) Weights() []*mat.Dense {
	return make([]*mat.Dense, 0)
}

func newRandomMask(size int, p float64) *mat.Dense {
	vals := make([]float64, size)

	for i := range vals {
		if rand.Float64() > p {
			vals[i] = 1
		}
	}

	return mat.NewDense(1, size, vals)
}
