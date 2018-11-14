package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const machineEpsilon = 2.2e-16

// NumericGradient computes a numerical approximation to ∇_x f.
// Returns a matrix with numerically approximated entries ∂f/∂x_{i,j}.
func NumericGradient(x *mat.Dense, f func(*mat.Dense) float64) *mat.Dense {
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
