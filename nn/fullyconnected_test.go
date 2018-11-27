package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestFullyConnected(t *testing.T) {
	x := mat.NewDense(3, 4, []float64{
		13, 2.01, -1, 3.0,
		-3, 12.45, 5.4, 9.1,
		30, 40.12, -50, 4.5,
	})

	y := mat.NewDense(3, 3, []float64{
		30, -1.0, 15.3,
		0.23, 34, 5.4,
		4, 2.343, -0.3333,
	})

	l := NewFullyConnectedLayer(4, 3)
	l.UpdateWeights = false

	err := SimpleGradientTest(l, x, y)
	assert.NoError(t, err)
}

func TestFullyConnectedXGrad(t *testing.T) {
	x := mat.NewDense(3, 4, []float64{
		13, 2.01, -1, 3.0,
		-3, 12.45, 5.4, 9.1,
		30, 40.12, -50, 4.5,
	})

	y := mat.NewDense(3, 3, []float64{
		30, -1.0, 15.3,
		0.23, 34, 5.4,
		4, 2.343, -0.3333,
	})

	w := mat.NewDense(4, 3, []float64{
		13, 2.01, -1,
		-3, 43.45, 5.4,
		30, 40.12, -50,
		2, 4, 3,
	})

	b := mat.NewDense(1, 3, []float64{
		0.1,
		0.4,
		-0.2,
	})

	stubTestXGrad := &ValueStub{
		ForwardsImpl: func(x *mat.Dense) *mat.Dense {
			return fullyConnectedForwards(x, w, b)
		},
		BackwardsImpl: func(grad *mat.Dense) *mat.Dense {
			xGrad, _, _ := fullyConnectedBackwards(grad, x, w, b)
			return xGrad
		},
	}

	err := SimpleGradientTest(stubTestXGrad, x, y)
	assert.NoError(t, err)
}

func TestFullyConnectedWGrad(t *testing.T) {
	x := mat.NewDense(3, 4, []float64{
		13, 2.01, -1, 3.0,
		-3, -40.45, -5.4, 9.1,
		30, -40.12, -50, 4.5,
	})

	y := mat.NewDense(3, 3, []float64{
		30, -1.0, 15.3,
		0.23, 34, 5.4,
		4, -2.343, -0.3333,
	})

	w := mat.NewDense(4, 3, []float64{
		13, 2.01, -1,
		-3, -41.45, 5.4,
		30, 40.12, -50,
		2, 4, 3,
	})

	b := mat.NewDense(1, 3, []float64{
		0.1,
		0.4,
		-200.0,
	})

	stubTestXGrad := &ValueStub{
		ForwardsImpl: func(w *mat.Dense) *mat.Dense {
			return fullyConnectedForwards(x, w, b)
		},
		BackwardsImpl: func(grad *mat.Dense) *mat.Dense {
			_, wGrad, _ := fullyConnectedBackwards(grad, x, w, b)
			return wGrad
		},
	}

	err := SimpleGradientTest(stubTestXGrad, w, y)
	assert.NoError(t, err)
}

// ValueStub makes it possible to test arbitrary forwards an backwards
// implementations.
type ValueStub struct {
	ForwardsImpl, BackwardsImpl func(X *mat.Dense) *mat.Dense
}

func (v *ValueStub) Forwards(x *mat.Dense) *mat.Dense {
	return v.ForwardsImpl(x)
}

func (v *ValueStub) Backwards(x *mat.Dense) *mat.Dense {
	return v.BackwardsImpl(x)
}
