package sgd

import (
	"errors"
	"fmt"
	"log"
	"math/rand"

	"github.com/rosshemsley/gonn/nn"
	"gonum.org/v1/gonum/mat"
)

type Setting func(*Config)

type Config struct {
	numEpochs int
	batchSize int
}

type LossFunction func(X, Y *mat.Dense) *mat.Dense

func SGD(x, y *mat.Dense, loss nn.Loss, net nn.Value, settings ...Setting) {
	cfg := initConfig(settings...)

	for epoch := 0; epoch < cfg.numEpochs; epoch++ {
		xBatches, yBatches := createShuffledBatches(x, y, cfg.batchSize)

		for i := range xBatches {
			xBatch, yBatch := xBatches[i], yBatches[i]
			yHat := net.Forwards(xBatch)
			l, grad := loss(yBatch, yHat)
			L2Regularize(net)
			net.Backwards(grad)
			log.Printf("Loss for batch %d/%d, epoch %d/%d: %f", i, len(xBatches), epoch, cfg.numEpochs, l)
		}

	}
}

func L2Regularize(v nn.Value) {
	regularizationConstant := 0.001
	for _, w := range v.Weights() {
		deltaW := mat.DenseCopyOf(w)
		deltaW.Scale(float64(-nn.LearningRate*regularizationConstant), deltaW)
		w.Add(w, deltaW)
	}
}

func WithBatchSize(n int) Setting {
	return func(c *Config) {
		c.batchSize = n
	}
}

func WithEpochs(n int) Setting {
	return func(c *Config) {
		c.numEpochs = n
	}
}

func createShuffledBatches(x *mat.Dense, y *mat.Dense, batchSize int) ([]*mat.Dense, []*mat.Dense) {
	xRows, xCols := x.Dims()
	yRows, yCols := y.Dims()
	if xRows != yRows {
		panic(fmt.Sprintf("mismatch in dimensions: %d != %d", xRows, yRows))
	}

	xs := make([][]float64, xRows)
	ys := make([][]float64, yRows)

	for i := 0; i < xRows; i++ {
		xs[i] = x.RawRowView(i)
		ys[i] = y.RawRowView(i)
	}

	rand.Shuffle(len(xs), func(i, j int) {
		xs[i], xs[j] = xs[j], xs[i]
		ys[i], ys[j] = ys[j], ys[i]
	})

	xBatches := make([]*mat.Dense, 0)
	yBatches := make([]*mat.Dense, 0)

	xCurrent := make([]float64, 0)
	yCurrent := make([]float64, 0)
	currentCount := 0

	for i := 0; i < xRows; i++ {
		if currentCount == batchSize || i == xRows-1 {
			xBatches = append(xBatches, mat.NewDense(currentCount, xCols, xCurrent))
			yBatches = append(yBatches, mat.NewDense(currentCount, yCols, yCurrent))

			xCurrent = make([]float64, 0)
			yCurrent = make([]float64, 0)
			currentCount = 0
		}

		xCurrent = append(xCurrent, xs[i]...)
		yCurrent = append(yCurrent, ys[i]...)
		currentCount++
	}

	xVals := make([]float64, 0)
	yVals := make([]float64, 0)
	for i := range xs {
		xVals = append(xVals, xs[i]...)
		yVals = append(yVals, ys[i]...)
	}

	return xBatches, yBatches
}

func sampleBatch(X, Y *mat.Dense, n int) (XSample, YSample *mat.Dense, err error) {
	xNRows, xNCols := X.Dims()
	if n > xNRows {
		return nil, nil, errors.New("training data must exceed batch size")
	}

	_, yNCols := Y.Dims()

	indices := make([]int, 0, n)
	for len(indices) < n {
		indices = append(indices, rand.Intn(xNRows))
	}

	XSample = mat.NewDense(n, xNCols, nil)
	YSample = mat.NewDense(n, yNCols, nil)

	for i := 0; i < n; i++ {
		XSample.SetRow(i, X.RawRowView(i))
		YSample.SetRow(i, Y.RawRowView(i))
	}

	return
}

func initConfig(settings ...Setting) Config {
	cfg := Config{
		numEpochs: 1,
		batchSize: 1,
	}
	for _, s := range settings {
		s(&cfg)
	}
	return cfg
}
