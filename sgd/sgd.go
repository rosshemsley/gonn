package sgd

import (
	"errors"
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
		xBatch, yBatch, _ := sampleBatch(x, y, cfg.batchSize)

		yHat := net.Forwards(xBatch)
		l, grad := loss(yBatch, yHat)
		L2Regularize(net)

		net.Backwards(grad)

		log.Printf("Loss at step %d: %f", epoch, l)
	}
}

func L2Regularize(v nn.Value) {
	regularizationConstant := 0.01
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
