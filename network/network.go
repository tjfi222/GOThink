package Network

//
// Update: June 10th, 2018
// The "Network" package is the highest level abstraction of our neural network model.
// It is dependant on only 2 external packages, the "Neuron" and "Layer" packages. 
// A neural network can be defined by 9 variables defined in the "Network" struct defined below.
// This package also include the Forward() function to be able to run the model on a set of inputs.

import (
	N "../neuron"
	L "../layer"
)

//
// Network struct defines the entire architecture of the neural network. 
// 

type Network struct {
	Inputs int
	Outputs int
	Hidden int
	Activation int
	Lrate float64
	Mom float64
	Bias float64
	HiddenLayers []L.Layer
	OutputLayer L.Layer
}

//
// The Forward function accepts the neural network input array "x" and the network struct "net" that it will propogate forward.
//
func Forward(x []float64, net Network) []float64{
	// The output of the nerual network defined as "out"
	out := []float64{}

	// Loop through the hidden layers of the network.
	for h := range net.HiddenLayers {

		// Loop through each neuron in the hidden layer
		for i := range net.HiddenLayers[h].Neurons {
			// Initialize the output of the neuron 
			value := 0.0
			// Loop through the weights of the neuron
			for j := range net.HiddenLayers[h].Neurons[i].Weights {
				// If this is the first layer (h = 0) then use the input array "x". Else use the previous hidden layer output.
				if h == 0 {
					value += net.HiddenLayers[h].Neurons[i].Weights[j]*x[j]					
				} else {
					// Multiply the currnt neuron weight by the output of the connected input neuron. 
					value += net.HiddenLayers[h].Neurons[i].Weights[j]*net.HiddenLayers[h-1].Neurons[j].Output				
				}
			}
			// Update the "value" variale with the activated version of the neuron output
			value = N.Activate(value + net.HiddenLayers[h].Neurons[i].Bias, net.HiddenLayers[h].Neurons[i].Activation)
			net.HiddenLayers[h].Neurons[i].Output = value
		}
	}

	// The output layer is handled slightly differently. 
	endLayers := len(net.HiddenLayers) - 1

	for i := range net.OutputLayer.Neurons {
		value := 0.0
		for j := range net.OutputLayer.Neurons[i].Weights {
			value += net.HiddenLayers[endLayers].Neurons[j].Output * net.OutputLayer.Neurons[i].Weights[j]
		}
		value = N.Activate(value + net.OutputLayer.Neurons[i].Bias, net.HiddenLayers[0].Neurons[0].Activation)
		net.OutputLayer.Neurons[i].Output = value
		out = append(out, value)
	}
	return out
}

//
// "NewNetwork" returns a new network struct with defined parameters.
// It also creates new layers and new neurons for each layer to initialize the entire network.
//
func NewNetwork(x int, n float64, a float64, act int, b float64, h int, r int, o int) Network{
	net := Network{Lrate: n, Mom: a, Bias: b, Activation: act, Inputs: x, Outputs: o, Hidden: h}
	for i := 0; i < r; i++ {
		if i == 0 {
			l := L.NewHidden(x, n, a, act, b, h)
			net.HiddenLayers = append(net.HiddenLayers, l)			
		} else {
			l := L.NewHidden(h, n, a, act, b, h)	
			net.HiddenLayers = append(net.HiddenLayers, l)			
		}
	}
	net.OutputLayer = L.NewOutput(h, n, a, act, b, o)

	return net
}