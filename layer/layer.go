package Layer

import (
	N "../neuron"
	"math/rand"
)

type Layer struct {
	Neurons []N.Neuron
	Lrate float64
	Mom float64
	Inputs []float64
	Outputs []float64
}


func NewHidden(x int, n float64, a float64, act int, b float64, z int) Layer{
	l := Layer{Lrate: n, Mom: a}
	for i := 0; i < z; i++ {
		b := rand.Float64()*(2)-1
		nu := N.Neuron{Inputs: x, Activation: act, Bias: b}
		nu = N.InitialWeights(nu)
		l.Neurons = append(l.Neurons, nu)
	}
	return l
}

func NewOutput(x int, n float64, a float64, act int, b float64, o int) Layer{
	l := Layer{Lrate: n, Mom: a}
	for i := 0; i < o; i++ {
		b := rand.Float64()*(2)-1
		n := N.Neuron{Inputs: x, Activation: act, Bias: b}
		n = N.InitialWeights(n)
		l.Neurons = append(l.Neurons, n)
	}
	return l
}