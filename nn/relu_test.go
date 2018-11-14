package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestRelu(t *testing.T) {
	x := mat.NewDense(3, 3, []float64{
		1, 2, -1,
		-3, 4, 5,
		30, 40, -50,
	})

	expected := mat.NewDense(3, 3, []float64{
		1, 2, 0,
		0, 4, 5,
		30, 40, 0,
	})

	r := relu(x)

	delta := mat.NewDense(3, 3, nil)
	delta.Sub(r, expected)
	if norm(delta) > 0.001 {
		t.Errorf("unexpected relu value: %v != %v", r, expected)
	}

}

func TestReluBackwards(t *testing.T) {
	x := mat.NewDense(3, 3, []float64{
		1, 2, -1,
		-3, 4, 5,
		30, 40, -50,
	})

	g := mat.NewDense(3, 3, []float64{
		3, 4, 5,
		6, 7, 8,
		9, 10, 11,
	})

	expected := mat.NewDense(3, 3, []float64{
		3, 4, 0,
		0, 7, 8,
		9, 10, 0,
	})

	r := reluBackwards(g, x)

	delta := mat.NewDense(3, 3, nil)
	delta.Sub(r, expected)
	if norm(delta) > 0.001 {
		t.Errorf("unexpected relu value: %v != %v", r, expected)
	}
}

func TestReluNumericGradient(t *testing.T) {
	x := mat.NewDense(3, 3, []float64{
		13, 2.01, -1,
		-3, 410.45, 5.4,
		30, 40.12, -50,
	})

	y := mat.NewDense(3, 3, []float64{
		30, -1.0, 15.3,
		0.23, 34, 5.4,
		4, 2.343, -0.3333,
	})

	err := SimpleGradientTest(NewRelu(), x, y)
	assert.NoError(t, err)
}
