package nn

import (
	"gonum.org/v1/gonum/mat"
)

const LearningRate = 0.2

// Value represents any node in an NN computation graph.
type Value interface {
	Forwards(X *mat.Dense) *mat.Dense
	Backwards(X *mat.Dense) *mat.Dense

	// SetTrainingEnabled tells this node whether not to store gradients and perform dropout.
	SetTrainingEnabled(bool)

	// Weights returns a slice of all variables that are updated during training.
	// This makes it possible to implement regularization.
	Weights() []*mat.Dense
}

type Loss func(yHat *mat.Dense, y *mat.Dense) (loss float64, grad *mat.Dense)
