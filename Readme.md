# 🧠 Gonn - Go Neural Network

##### Because there aren't enough half-baked neural network libraries in the world yet.

A bare-bones deep neural network written in Go.
Contains just enough code to train a simple network and make predictions.

Uses `gonum` for linear algebra.

## Example

```go    
// Load up a training set
x, err := mnist.LoadImagesGzipFile("data/train-images-idx3-ubyte.gz")
if err != nil {
    log.Fatalf("Failed to load images: %s", err)
}
y, err := mnist.LoadLabelsGzipFile("data/train-labels-idx1-ubyte.gz")
if err != nil {
    log.Fatalf("Failed to load labels: %s", err)
}

_, cols := x.Dims()

dnn := nn.NewFeedForwardNetwork(
    nn.NewFullyConnectedLayer(cols, 10),
    nn.NewSoftMaxLayer(),
)

sgd.SGD(x, y, nn.L2Loss, dnn, sgd.WithBatchSize(256), sgd.WithEpochs(10))
```