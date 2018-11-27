package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNumericGradient(t *testing.T) {
	f := func(x *mat.Dense) float64 {
		return x.At(0, 0) * x.At(0, 0)
	}

	grad := NumericGradient(f, mat.NewDense(1, 1, []float64{10}))
	assert.InDelta(t, grad.At(0, 0), 20.0, 0.1)
}

type noopValue struct{}

func (noopValue) Forwards(x *mat.Dense) *mat.Dense {
	return x
}

func (noopValue) Backwards(y *mat.Dense) *mat.Dense {
	return y
}
func (noopValue) Weights() []*mat.Dense {
	return make([]*mat.Dense, 0)
}

func TestSimpleGradientTest(t *testing.T) {
	x := mat.NewDense(3, 3, []float64{
		13, 2.01, -1,
		-3, 410.45, 5.4,
		30, 40.12, -50,
	})

	y := mat.NewDense(3, 3, []float64{
		30, -1.0, 15.3,
		0.23, 34, 5.4,
		4, 2.343, -0.3333,
	})

	assert.NoError(t, SimpleGradientTest(noopValue{}, x, y))
}
