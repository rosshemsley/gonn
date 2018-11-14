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

	grad := NumericGradient(mat.NewDense(1, 1, []float64{10}), f)
	assert.InDelta(t, grad.At(0, 0), 20.0, 0.1)
}
