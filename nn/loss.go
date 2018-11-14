package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func L2Loss(yHat *mat.Dense, y *mat.Dense) (float64, *mat.Dense) {
	rows, cols := yHat.Dims()
	l := l2(y, yHat)

	grad := mat.NewDense(rows, cols, nil)
	// TODO(Ross): I think this is wrong.
	grad.Sub(yHat, y)
	return l / float64(rows), grad
}

func l2(x, y *mat.Dense) float64 {
	rows, cols := x.Dims()

	var l float64
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			d := y.At(r, c) - x.At(r, c)
			l += math.Pow(d, 2)
		}
	}

	return l
}
