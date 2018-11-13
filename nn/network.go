package nn

import (
	"gonum.org/v1/gonum/mat"
)

// FeedForwardNetwork is a simple feed forward network.
type FeedForwardNetwork struct {
	layers []Value
}

func NewFeedForwardNetwork(layers ...Value) *FeedForwardNetwork {
	return &FeedForwardNetwork{
		layers: layers,
	}
}

// Forwards pushes values through network.
func (n *FeedForwardNetwork) Forwards(x *mat.Dense) *mat.Dense {
	var v = x

	for _, layer := range n.layers {
		v = layer.Forwards(v)
	}

	return v
}

// Backwards flows the gradient back through the network.
func (n *FeedForwardNetwork) Backwards(x *mat.Dense) *mat.Dense {
	v := x

	for i := len(n.layers) - 1; i >= 0; i-- {
		v = n.layers[i].Backwards(v)
	}

	return v
}
