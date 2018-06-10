package main

import (
	"./prep"
	NW "./network"
	T "./trainer"
	"math/rand"
	"time"
	"fmt"
	"os"
	"io/ioutil"
	"strings"
	"strconv"

)

func findMax(x []float64) float64{
	y := 0.0
	for i := range x {
		if x[i] > y {
			y = x[i]
		}
	}
	return y
}

func findMax2(x []float64) int{
	y := -100.0
	var z int
	for i := range x {
		if x[i] > y {
			y = x[i]
			z = i
		}
	}
	return z
}

func myWrite(n string, x [][]float64) {
	outFile, err := os.Create(n)
	if err != nil {
		fmt.Println(err)
	}
	defer outFile.Close()

	for i := range x {
		row := []string{}
		for j := range x[i] {
			row = append(row, strconv.FormatFloat(x[i][j], 'f', -1,  64))
		}
		fmt.Fprintf(outFile, strings.Join(row, ","))
		if i != (len(x) - 1) {
			fmt.Fprintf(outFile, "\n")			
		}

	}

}

func myWrite2(n string, x [10][10]float64) {
	outFile, err := os.Create(n)
	if err != nil {
		fmt.Println(err)
	}
	defer outFile.Close()

	for i := range x {
		row := []string{}
		for j := range x[i] {
			row = append(row, strconv.FormatFloat(x[i][j], 'f', -1,  64))
		}
		fmt.Fprintf(outFile, strings.Join(row, ","))
		if i != (len(x) - 1) {
			fmt.Fprintf(outFile, "\n")			
		}

	}

}

func main() {

	rand.Seed(time.Now().UTC().UnixNano())
	

	dataFile, err := ioutil.ReadFile("MNISTnumImages5000.txt")
	targetFile, err := ioutil.ReadFile("MNISTnumLabels5000.txt")
	
	if err != nil {
		fmt.Println(err)
	}
	lines := strings.Split(string(dataFile), "\n")
	tLines := strings.Split(string(targetFile), "\n")
	tLines = tLines[:len(tLines)-1]
	targets := []int{}
	for i := range tLines {
		value, _ := strconv.Atoi(tLines[i])
		targets = append(targets, value)
	}

	rawData := [][]string{}
	offset := 0
	for i := range lines {
		if i > (offset -1) {
			line := strings.Split(lines[i], " ")
			rawData = append(rawData, line)
		}
	}

	cleanData := [][]float64{}
	rawData = rawData[:len(rawData)-1]
	fmt.Println(len(rawData))
	for i := range rawData {
		value1 := strings.Fields(rawData[i][0])
		value2 := []float64{}
		for j := range value1 {
			value3, _ := strconv.ParseFloat(value1[j],64)
			value2 = append(value2, value3)
		}

		cleanData = append(cleanData, value2)
	}



	// Split Data
	testD, trainD, testT, trainT := Prep.SplitData(cleanData, targets, 20)



	// Define Neural Network Structure and Parameters
	net := NW.NewNetwork(784, 0.5, 0.03, 0, 0.0, 200, 1, 10)
	

	h := NW.Forward(testD[0], net)
	myError := T.CalcErrors(h, trainT[0])
	totalError := 0.0
	hits := 0.0
	count := 0.0
	trainIndex := []int{}

	confuseTrain := [10][10]float64{}
	confuseTest := [10][10]float64{}

	hr := [][]float64{}
	hit_rates := []float64{}
	trainRate := 0.0
	testRate := 0.0

	for i := 0; i < 500; i++ {
		if i % 10 == 0 {
			fmt.Println("Epoch ", i)
			hits = 0.0
			count = 0.0
			for j := range testD {
				h := NW.Forward(testD[j], net)
				for z := range h {
					if z == testT[j] {
						myMax := findMax2(h)
						if z == myMax {
							hits += 1
							
						}
					}   
				}
				count += 1
			}

			fmt.Println(hits / 1000)	
			testRate = hits / 1000

			hits = 0
			count = 0.0
			for j := range trainD {
				h := NW.Forward(trainD[j], net)
				for z := range h {
					if z == trainT[j] {
						myMax := findMax2(h)
						if z == myMax {
							hits += 1
						}
					}   
				}
				count += 1
			}

			fmt.Println(hits / 4000)
			trainRate = (hits/ 4000)


			hit_rates := []float64{testRate, trainRate}
			hr = append(hr, hit_rates)
		}

		trainIndex = Prep.RandArray(len(trainD))
		
		for j := range trainD {
			target := [1]float64{float64(trainT[trainIndex[j]])}
			net = T.Backward(trainD[trainIndex[j]], net, target[:], true)

		}
		
		totalError = 0.0
		h := NW.Forward(trainD[0], net)

		myError = T.CalcErrors(h, trainT[0])
		for j := range myError {
			totalError += myError[j]
		}

	}




	// Document

	hits = 0.0
	count = 0.0
	for j := range testD {
		h := NW.Forward(testD[j], net)
		for z := range h {
			if z == testT[j] {
				otherMax := findMax2(h)

				confuseTest[z][otherMax] += 1
				if z == otherMax {
					hits += 1
				}
			}   
		}
		count += 1
	}

	fmt.Println(hits / 1000)	
	testRate = hits / 1000

	hits = 0.0
	count = 0.0
	for j := range trainD {
		h := NW.Forward(trainD[j], net)
		for z := range h {
			if z == trainT[j] {
				otherMax := findMax2(h)
				confuseTrain[z][otherMax] += 1
				if z == otherMax {
					hits += 1
				}
			}   
		}
		count += 1
	}

	fmt.Println(hits / 4000)
	trainRate = (hits/ 4000)

	hit_rates = []float64{testRate, trainRate}
	hr = append(hr, hit_rates)


	fmt.Println("Saving Weights...")
	hweights := [][]float64{}
	for i := range net.HiddenLayers {
		temp := []float64{}
		for j := range net.HiddenLayers[i].Neurons {
			for z := range net.HiddenLayers[i].Neurons[j].Weights {
				temp = append(temp, net.HiddenLayers[i].Neurons[j].Weights[z])
			}
			temp = append(temp, net.HiddenLayers[i].Neurons[j].Bias)
		}
		hweights = append(hweights, temp)
	}
	
	myWrite("final_weights.txt", hweights)
	fmt.Println("Done!")



	myWrite("MINST_Performance.txt", hr)
	myWrite2("ConfuseTrain.txt", confuseTrain)
	myWrite2("ConfuseTest.txt", confuseTest)


	fmt.Println(myError)


	
	// This stuff is not important a all
	fmt.Println(len(testD))
	fmt.Println(len(trainD[0]))
	fmt.Println(len(testT))
	fmt.Println(len(trainT))
	fmt.Println(len(net.HiddenLayers[0].Neurons))
	fmt.Println(len(h))
	fmt.Println(len(net.OutputLayer.Neurons))
	fmt.Println(net.OutputLayer.Neurons[0].Weights[0])
	

}
