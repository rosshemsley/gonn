package nn

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

const machineEpsilon = 2.2e-16

// x := mat.NewDense(2, 3, []float64{
// 	118.0, 14.3,
// 	0.01, -12.5,
// 	-13.0, 2.0,
// })

// SimpleGradientTest connects the given value to the L2 loss, and measure the
// gradient as computed as the loss against the given test point y.
func SimpleGradientTest(v Value, x, y *mat.Dense) error {
	var gradLoss *mat.Dense

	_, gradLoss = L2Loss(v.Forwards(x), y)
	gradAnalytic := v.Backwards(gradLoss)

	f := func(x *mat.Dense) float64 {
		l, _ := L2Loss(v.Forwards(x), y)
		return l
	}

	gradNumeric := NumericGradient(f, x)
	rows, cols := gradNumeric.Dims()

	delta := mat.NewDense(rows, cols, nil)
	delta.Sub(gradAnalytic, gradNumeric)
	if norm(delta) > 1e-3 {
		return fmt.Errorf("expected: %v, actual: %v", gradNumeric, gradAnalytic)
	}

	return nil
}

// NumericGradient computes a numerical approximation to ∇_x f.
// Returns a matrix with numerically approximated entries ∂f/∂x_{i,j}.
func NumericGradient(f func(*mat.Dense) float64, x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			xTmp := mat.DenseCopyOf(x)
			v := xTmp.At(i, j)
			h := chooseH(v)

			xTmp.Set(i, j, v-h)
			y1 := f(xTmp)
			xTmp.Set(i, j, v+h)
			y2 := f(xTmp)

			result.Set(i, j, (y2-y1)/(2*h))
		}
	}

	return result
}

// Choose the step to use. This was taken from Wikipedia with almost no care.
func chooseH(x float64) float64 {
	return math.Sqrt(machineEpsilon) * x
}
