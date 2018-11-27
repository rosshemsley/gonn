package nn

import (
	"gonum.org/v1/gonum/mat"
)

type FullyConnectedLayer struct {
	// UpdateWeights controls whether or not the weights are updated on calling Backwards.
	// Defaults to true.
	UpdateWeights bool

	w          *mat.Dense
	b          *mat.Dense
	x          *mat.Dense
	activation Value

	trainingEnabled bool
}

func NewFullyConnectedLayer(inputDimension, outputDimension int) *FullyConnectedLayer {
	l := &FullyConnectedLayer{
		w:               NewRandomMatrix(inputDimension, outputDimension),
		b:               NewRandomMatrix(1, outputDimension),
		activation:      NewRelu(),
		trainingEnabled: true,
		UpdateWeights:   true,
	}

	return l
}

func (l *FullyConnectedLayer) SetTrainingEnabled(b bool) {
	l.trainingEnabled = b
}

func (l *FullyConnectedLayer) Forwards(x *mat.Dense) *mat.Dense {
	l.x = x
	a := fullyConnectedForwards(x, l.w, l.b)
	return l.activation.Forwards(a)
}

func (l *FullyConnectedLayer) Backwards(grad *mat.Dense) *mat.Dense {
	grad = l.activation.Backwards(grad)
	result, deltaW, deltaB := fullyConnectedBackwards(grad, l.x, l.w, l.b)

	if l.UpdateWeights {
		deltaW.Scale(-LearningRate, deltaW)
		deltaB.Scale(-LearningRate, deltaB)

		l.w.Add(l.w, deltaW)
		l.b.Add(l.b, deltaB)
	}

	return result
}

func (l *FullyConnectedLayer) Weights() []*mat.Dense {
	// Note(Ross): this weights slice is used for regularization.
	// going wisdom is that the bias term doesn't need to be included.
	return []*mat.Dense{l.w}
}

func fullyConnectedForwards(x, w, b *mat.Dense) *mat.Dense {
	xRows, _ := x.Dims()
	_, wCols := w.Dims()
	output := mat.NewDense(xRows, wCols, nil)
	oRows, oCols := output.Dims()
	output.Mul(x, w)

	// TODO(Ross): surely there's a nicer way to do this.
	for r := 0; r < oRows; r++ {
		for c := 0; c < oCols; c++ {
			output.Set(r, c, output.At(r, c)+b.At(0, c))
		}
	}

	return output
}

func fullyConnectedBackwards(grad, x, w, b *mat.Dense) (xGrad, wGrad, bGrad *mat.Dense) {
	rows, cols := grad.Dims()
	bGrad = mat.NewDense(1, cols, nil)

	for r := 0; r < rows; r++ {
		row := mat.NewDense(1, cols, grad.RawRowView(r))
		bGrad.Add(bGrad, row)
	}

	wRows, wCols := w.Dims()
	wGrad = mat.NewDense(wRows, wCols, nil)
	wGrad.Mul(x.T(), grad)

	xRows, xCols := x.Dims()
	xGrad = mat.NewDense(xRows, xCols, nil)
	xGrad.Mul(grad, w.T())
	return
}
