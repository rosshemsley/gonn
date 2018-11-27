package nn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

// func TestLossValue(t *testing.T) {
// 	yHat := mat.NewDense(1, 3, []float64{
// 		0.5,
// 		0.3,
// 		0.2,
// 	})

// 	y := mat.NewDense(1, 3, []float64{
// 		1,
// 		0,
// 		0,
// 	})

// 	l, grad := L2Loss(y, yHat)
// 	log.Printf("loss: %v, grad: %v", l, grad)
// 	t.Fail()

// }

func TestL2Loss(t *testing.T) {
	y := mat.NewDense(2, 3, []float64{
		1.0, 3.0,
		5.0, 7.0,
		11.0, 13.0,
	})

	var gradAnalytic *mat.Dense

	f := func(x *mat.Dense) (l float64) {
		l, gradAnalytic = L2Loss(y, x)
		return l
	}

	x := mat.NewDense(2, 3, []float64{
		118.0, 14.3,
		0.01, -12.5,
		-13.0, 2.0,
	})

	gradNumeric := NumericGradient(f, x)

	delta := mat.NewDense(2, 3, nil)
	delta.Sub(gradAnalytic, gradNumeric)
	if norm(delta) > 1e-3 {
		t.Errorf("expected gradient and actual gradient for l2 loss do not agree: %v, %v", gradAnalytic, gradNumeric)
	}
}
