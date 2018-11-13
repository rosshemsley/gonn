package nn

import (
	"gonum.org/v1/gonum/mat"
)

type FullyConnectedLayer struct {
	w          *mat.Dense
	b          *mat.Dense
	x          *mat.Dense
	activation Value
}

func NewFullyConnectedLayer(inputDimension, outputDimension int) *FullyConnectedLayer {
	return &FullyConnectedLayer{
		w:          newRandomMatrix(inputDimension, outputDimension),
		b:          newRandomMatrix(1, outputDimension),
		activation: NewRelu(outputDimension),
	}
}

func (l *FullyConnectedLayer) Forwards(x *mat.Dense) *mat.Dense {
	l.x = x

	xRows, _ := x.Dims()
	_, wCols := l.w.Dims()
	output := mat.NewDense(xRows, wCols, nil)
	oRows, oCols := output.Dims()
	output.Mul(x, l.w)

	// TODO(Ross): surely there's a nicer way to do this.
	for r := 0; r < oRows; r++ {
		for c := 0; c < oCols; c++ {
			output.Set(r, c, output.At(r, c)+l.b.At(0, c))
		}
	}

	return l.activation.Forwards(output)
}

func (l *FullyConnectedLayer) Backwards(grad *mat.Dense) *mat.Dense {
	grad = l.activation.Backwards(grad)

	rows, cols := grad.Dims()
	bGrad := mat.NewDense(1, cols, nil)

	for r := 0; r < rows; r++ {
		row := mat.NewDense(1, cols, grad.RawRowView(r))
		bGrad.Add(bGrad, row)
	}

	// Update b with ∇_b(L)

	bGrad.Scale(LearningRate, bGrad)
	// log.Printf("Update b: %v", bGrad)
	l.b.Sub(l.b, bGrad)

	// Compute ∇_W(L)
	// gRows, gCols := grad.Dims()
	// xRows, xCols := l.x.Dims()
	wRows, wCols := l.w.Dims()
	wGrad := mat.NewDense(wRows, wCols, nil)
	// log.Printf("HERE: ∇: %d, %d  x: %d, %d  w: %d, %d", gRows, gCols, xRows, xCols, wRows, wCols)
	wGrad.Mul(l.x.T(), grad)

	wGrad.Scale(LearningRate, wGrad)
	l.w.Sub(l.w, wGrad)

	// Compute ∇_x(L) and return it
	// gRows, gCols := grad.Dims()
	xRows, xCols := l.x.Dims()
	// log.Printf("∇: %d, %d. w: %d, %d x: %d, %d", gRows, gCols, wRows, wCols, xRows, xCols)
	result := mat.NewDense(xRows, xCols, nil)
	result.Mul(grad, l.w.T())
	return result
	// return nil
}

## Examples

This project uses go modules. If you have go1.11 or above, you can try this out by running

```
go get github.com/rosshemsley/gonn
```

If you have your go bin in your bath, you can now try the examples with 

```
$ gonn <example name>
```