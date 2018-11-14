package nn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestL2Loss(t *testing.T) {
	yHat := mat.NewDense(2, 3, []float64{
		1.0, 3.0,
		5.0, 7.0,
		11.0, 13.0,
	})

	var gradAnalytic *mat.Dense

	f := func(x *mat.Dense) (l float64) {
		l, gradAnalytic = L2Loss(x, yHat)
		return l
	}

	x := mat.NewDense(2, 3, []float64{
		118.0, 14.3,
		0.01, -12.5,
		-13.0, 2.0,
	})

	gradNumeric := NumericGradient(x, f)

	delta := mat.NewDense(2, 3, nil)
	delta.Sub(gradAnalytic, gradNumeric)
	if norm(delta) > 1e-3 {
		t.Errorf("expected gradient and actual gradient for l2 loss do not agree: %v, %v", gradAnalytic, gradNumeric)
	}
}
