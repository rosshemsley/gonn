package nn

import "gonum.org/v1/gonum/mat"

func norm(m *mat.Dense) float64 {
	sum := 0.0

	m.Apply(func(_, _ int, v float64) float64 {
		sum += v * v
		return v
	}, m)

	return sum
}
