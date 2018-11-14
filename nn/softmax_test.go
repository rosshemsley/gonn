package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestSoftMax(t *testing.T) {
	x := mat.NewDense(3, 3, []float64{
		1, 2, 1,
		3, 4, 5,
		30, 40, 50,
	})

	s := NewSoftMaxLayer()
	y := s.Forwards(x)

	expected := mat.NewDense(3, 3, []float64{
		0.2119415576, 0.5761168848, 0.2119415576,
		0.09003057317, 0.2447284711, 0.6652409558,
		0.0000000020610, 0.00004539786861, 0.999954600,
	})

	delta := mat.NewDense(3, 3, nil)
	delta.Sub(expected, y)

	if norm(delta) > 0.01 {
		t.Errorf("Unexpected soft max values, got: %v, expected: %v", y, expected)
	}
}

func TestSoftMaxJacobian(t *testing.T) {
	sx := mat.NewDense(1, 3, []float64{
		0.09003057317, 0.2447284711, 0.6652409558,
	})

	j := softmaxJacobian(sx)

	expected := mat.NewDense(3, 3, []float64{
		0.09003057317 * (1 - 0.09003057317), -0.09003057317 * 0.2447284711, -0.09003057317 * 0.6652409558,
		-0.2447284711 * 0.09003057317, 0.2447284711 * (1 - 0.2447284711), -0.2447284711 * 0.6652409558,
		-0.6652409558 * 0.09003057317, -0.6652409558 * 0.2447284711, 0.6652409558 * (1 - 0.6652409558),
	})
	delta := mat.NewDense(3, 3, nil)
	delta.Sub(expected, j)

	if norm(delta) > 0.01 {
		t.Errorf("Unexpected soft max jacobian (delta: %f), got: %v, expected: %v", norm(delta), j, expected)
	}
}

func TestSoftmaxNumericGradient(t *testing.T) {
	x := mat.NewDense(3, 4, []float64{
		13, 2.01, -1,
		-3, 410.45, 5.4,
		30, 40.12, -50,
		12, 0.3, 1.0,
	})

	y := mat.NewDense(3, 4, []float64{
		30, -1.0, 15.3,
		0.23, 34, 5.4,
		4, 2.343, -0.3333,
		1, 2, 3,
	})

	err := SimpleGradientTest(NewSoftMaxLayer(), x, y)
	assert.NoError(t, err)
}
