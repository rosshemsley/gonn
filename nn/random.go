package nn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func NewRandomMatrix(r, c int) *mat.Dense {
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, rand.Float64()*2-1)
		}
	}

	return result
}
