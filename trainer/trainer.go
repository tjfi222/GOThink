package Trainer

import (

	NW "../network"
	"math"
)


func Backward(x []float64, net NW.Network, y []float64, cor bool) NW.Network {
	output := NW.Forward(x, net)


	// Calculate Output Neuron Deltas
	outputDeltas := []float64{}
	for i := range output {
		if cor {
			if float64(i) == y[0] {
				outputDeltas = append(outputDeltas, derOutput(0.75, output[i])*derSig(output[i]))							
			} else {
				outputDeltas = append(outputDeltas, derOutput(0.25, output[i])*derSig(output[i]))							
			}
		} else {
			outputDeltas = append(outputDeltas, derOutput(y[i], output[i])*derSig(output[i]))			
		}
	}

	// Calculate Hidden Neuron Deltas
	hiddenDeltas := [][]float64{}
	end := len(net.HiddenLayers) -1
	for h := range net.HiddenLayers {
		layerDeltas := []float64{}
		endLayer := len(hiddenDeltas) -1
		for i := range net.HiddenLayers[end-h].Neurons {
			dError := 0.0
			if h == 0 {
				for j := range net.OutputLayer.Neurons {
					dError += outputDeltas[j]*net.OutputLayer.Neurons[j].Weights[i]					
				}
			} else {
				for j := range net.HiddenLayers[end-h+1].Neurons {

					dError += hiddenDeltas[endLayer][j]*net.HiddenLayers[end-h+1].Neurons[j].Weights[i]					
				}
			}
			layerDeltas = append(layerDeltas, dError * derSig(net.HiddenLayers[end-h].Neurons[i].Output))
	
		}
		hiddenDeltas = append(hiddenDeltas, layerDeltas)
		
	}


	// Update Output Neuron Weights
	for i := range output {
		for j := range net.OutputLayer.Neurons[i].Weights {
			outputError := outputDeltas[i]*net.HiddenLayers[end].Neurons[j].Output
			net.OutputLayer.Neurons[i].Weights[j] += -net.OutputLayer.Lrate*outputError + net.OutputLayer.Mom*net.OutputLayer.Neurons[i].DeltaW[j]
			net.OutputLayer.Neurons[i].DeltaW[j] = outputError			
		}
		net.OutputLayer.Neurons[i].Bias += -net.OutputLayer.Lrate*outputDeltas[i]
		
	}


	// Update Hidden Weights
    for h := range net.HiddenLayers {
		for i := range net.HiddenLayers[end-h].Neurons {
			for j := range net.HiddenLayers[end-h].Neurons[i].Weights {
				hiddenError := 0.0
				if end-h == 0 {
					hiddenError = hiddenDeltas[h][i]*x[j]					
				} else {
					hiddenError = hiddenDeltas[h][i]*net.HiddenLayers[end-h-1].Neurons[j].Output										
				}
				net.HiddenLayers[end-h].Neurons[i].Weights[j] += -net.HiddenLayers[end-h].Lrate*hiddenError + net.HiddenLayers[end-h].Mom*net.HiddenLayers[end-h].Neurons[i].DeltaW[j]
				net.HiddenLayers[end-h].Neurons[i].DeltaW[j] = hiddenError
			}
			net.HiddenLayers[end-h].Neurons[i].Bias += -hiddenDeltas[h][i]*net.HiddenLayers[end-h].Lrate
			
		}
	}

	return net
}

func CalcErrors(x []float64, y int) []float64{
	errors := []float64{}
	for i := range x {
		if i == y {
			errors = append(errors, 0.5*math.Pow(1-x[i],2))
		} else {
			errors = append(errors, 0.5*math.Pow(0-x[i],2))
		}
	}
	return errors
}

func CalcLoss(t []float64, a []float64) float64 {
	myError := 0.0
	for i := range t {
		myError += 0.5*math.Pow(t[i]-a[i],2)
	}
	return myError
}

// The derivative of the output.
func derOutput(t float64, a float64) float64{
	return -(t - a)
}

// The derivative of the sigmoid function. 
func derSig(x float64) float64{
	return x * (1 - x)
}