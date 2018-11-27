package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SoftMax struct {
	sx *mat.Dense
}

func NewSoftMaxLayer() *SoftMax {
	return &SoftMax{}
}

func (s *SoftMax) Forwards(x *mat.Dense) *mat.Dense {
	s.sx = softmax(x)
	return softmax(x)
}

func (s *SoftMax) Backwards(grad *mat.Dense) *mat.Dense {
	result := make([]float64, 0)
	rows, cols := s.sx.Dims()
	_, gradCols := grad.Dims()

	for r := 0; r < rows; r++ {
		rowVals := s.sx.RawRowView(r)
		row := mat.NewDense(1, cols, rowVals)
		rowJacobian := softmaxJacobian(row)
		rowGrad := mat.NewDense(1, gradCols, grad.RawRowView(r))
		rowResultGrad := mat.NewDense(1, cols, nil)
		rowResultGrad.Mul(rowGrad, rowJacobian)
		result = append(result, rowResultGrad.RawMatrix().Data...)
	}

	return mat.NewDense(rows, cols, result)
}

func (s *SoftMax) Weights() []*mat.Dense {
	return make([]*mat.Dense, 0)
}

func softmaxLayerGradient(grad, x *mat.Dense) *mat.Dense {
	return nil
}

// softmax returns a numerically stable version of softmax that can deal with pretty big numbers.
func softmax(x *mat.Dense) *mat.Dense {
	xRows, xCols := x.Dims()

	result := mat.NewDense(xRows, xCols, nil)
	for r := 0; r < xRows; r++ {
		sum := 0.0

		max := x.At(r, 0)
		for c := 0; c < xCols; c++ {
			max = math.Max(max, x.At(r, c))
		}

		for c := 0; c < xCols; c++ {
			sum += math.Exp(x.At(r, c) - max)
		}
		for c := 0; c < xCols; c++ {
			result.Set(r, c, math.Exp(x.At(r, c)-max)/sum)
		}
	}

	return result
}

// Takes a vector s(x) and returns D S(x)
func softmaxJacobian(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	if rows != 1 {
		panic("softmax input has wrong shape")
	}
	result := mat.NewDense(cols, cols, nil)
	result.Mul(x.T(), x)
	result.Scale(-1, result)
	result.Add(result, mat.NewDiagonal(cols, x.RawRowView(0)))

	return result
}
