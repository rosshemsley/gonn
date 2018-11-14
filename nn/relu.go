package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Relu struct {
	x *mat.Dense
}

func NewRelu() *Relu {
	return &Relu{}
}

func (r *Relu) Forwards(x *mat.Dense) *mat.Dense {
	r.x = x
	return relu(x)
}

func (r *Relu) Backwards(grad *mat.Dense) *mat.Dense {
	return reluBackwards(grad, r.x)
}

func relu(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()

	result := mat.NewDense(rows, cols, nil)
	result.Apply(func(i, j int, v float64) float64 {
		return math.Max(v, 0)
	}, x)

	return result
}

func reluBackwards(grad, x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	result := mat.NewDense(rows, cols, nil)

	for ri := 0; ri < rows; ri++ {
		gradRow := mat.NewDense(1, cols, grad.RawRowView(ri))
		for i, v := range x.RawRowView(ri) {
			if v > 0 {
				result.Set(ri, i, gradRow.At(0, i))
			}
		}
	}

	return result
}
