package nn

import (
	"testing"

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
