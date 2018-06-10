package Neuron

// 
// Updated: June 10th 2018
// The "Neuron" package is dependant on just the "math" and "math/rand" packages.
// The package includes a struct to hold the Neuron variables and a function to initialize the weights as well as an activation function. 
// The activation and initial weight functions were included in the Neuron package in case a moe advanced model wanted to 
// alter the behavior of specific neurons. 
//

import (
	"math"
	"math/rand"
)

type Neuron struct {
	Inputs     int
	Activation int
	Weights    []float64
	DeltaW     []float64
	Bias       float64
	Output     float64
}

// For now the package only includes the sigmoid activation function but was written to be flexible in the future.
func Activate(x float64, act int) float64 {
	output := 0.0
	if act == 0 {
		output = 1 / (1 + math.Exp(-x))
	}
	return output
}

func InitialWeights(n Neuron) Neuron {
	for i := 0; i < n.Inputs; i++ {

		n.Weights = append(n.Weights, (rand.Float64()*(2))-1)

		n.DeltaW = append(n.DeltaW, 0.0)

	}
	return n
}