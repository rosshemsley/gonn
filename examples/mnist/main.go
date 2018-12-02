package mnist

import (
	"log"

	"github.com/rosshemsley/gonn/mnist"
	"github.com/rosshemsley/gonn/nn"
	"github.com/rosshemsley/gonn/sgd"
	"gonum.org/v1/gonum/mat"
)

func Run() {
	x, err := mnist.LoadImagesGzipFile("data/train-images-idx3-ubyte.gz")
	if err != nil {
		log.Fatalf("Failed to load images: %s", err)
	}
	_, xCols := x.Dims()

	y, err := mnist.LoadLabelsGzipFile("data/train-labels-idx1-ubyte.gz")
	if err != nil {
		log.Fatalf("Failed to load labels: %s", err)
	}

	dnn := nn.NewFeedForwardNetwork(
		nn.NewFullyConnectedLayer(xCols, 30),
		nn.NewDropoutLayer(0.05),
		nn.NewFullyConnectedLayer(30, 20),
		nn.NewDropoutLayer(0.05),
		nn.NewFullyConnectedLayer(20, 10),
		nn.NewSoftMaxLayer(),
	)

	log.Printf("Classification rate: %.2f%%", evaluate(dnn))
	startRate := evaluate(dnn)

	sgd.SGD(x, y, nn.L2Loss, dnn, sgd.WithBatchSize(64), sgd.WithEpochs(100))

	endRate := evaluate(dnn)
	log.Printf("Classification rate on test set: from %.2f%% to %.2f%%", startRate, endRate)
}

func evaluate(dnn nn.Value) float64 {
	x, err := mnist.LoadImagesGzipFile("data/t10k-images-idx3-ubyte.gz")
	if err != nil {
		log.Fatalf("Failed to load images: %s", err)
	}

	y, err := mnist.LoadLabelsGzipFile("data/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		log.Fatalf("Failed to load labels: %s", err)
	}

	rows, _ := x.Dims()

	correct := 0
	for i := 0; i < rows; i++ {
		img := extract(i, x)
		yHat := dnn.Forwards(img)

		classified := mnist.LabelValue(yHat.RawRowView(0))
		expected := mnist.LabelValue(extract(i, y).RawRowView(0))
		if classified == expected {
			correct++
		}
	}

	return float64(correct) / float64(rows) * 100.0
}

func extract(i int, x *mat.Dense) *mat.Dense {
	_, cols := x.Dims()
	vs := x.RawRowView(i)
	return mat.NewDense(1, cols, vs)
}
