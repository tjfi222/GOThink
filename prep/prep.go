package Prep

//
// Updated: June 10th 2018
// The "Prep" packages includes several functions that make repeated calculations easier and abracted away from other packages.
// 

import (
	"math/rand"
	"time"
)

func MinMax(data [][]float64, offset int) (float64, float64) {
	var min, max float64
	for i := range data {
		if i == 0 {
			min = data[i][offset]
			max = data[i][offset]
		} else {
			if data[i][offset] < min {
				min = data[i][offset]
			}
			if data[i][offset] > max {
				max = data[i][offset]
			}
		}
	}
	return min, max
}

func Nomralize(value, min, max float64) float64 {
	return (value - min) / (max - min)
}



func SplitData(data [][]float64, targets []int, test_percent float64) ([][]float64, [][]float64, []int, []int){
	rand.Seed(time.Now().UTC().UnixNano())
	ds := []int{}
	trainD :=[][]float64{}
	testD := [][]float64{}
	trainT :=[]int{}
	testT := []int{}
	index := 0
	found := false
	length := len(data)
	numRows := float64(len(data)) * (test_percent / 100)
	for i := 0; i < int(numRows); i++ {
		found = false
		index = rand.Intn(length)
		if findElem(ds, index) == false {
			ds = append(ds, index)
			found = true
		} else {
			for found == false {
				index = rand.Intn(length)
				if findElem(ds, index) == false {
					ds = append(ds, index)
					found = true
				}
			}
		}
	}
	for i := 0; i < length; i++ {
		if findElem(ds, i) == true {
			testD = append(testD, data[i])
			testT = append(testT, targets[i])			
		} else {
			trainD = append(trainD, data[i])
			trainT = append(trainT, targets[i])			
		}
	}

	return testD, trainD, testT, trainT
}

func findElem(data []int, value int) bool {
	for i := range data{
		if value == data[i]{
			return true
		}
	}
	return false
}

func RandArray(l int) []int {
	x := []int{}
	var y int
	var found bool
	for i := 0; i < l; i++ {
		found = false
		for found == false {
			y = rand.Intn(l)
			if findElem(x, y) == false {
				x = append(x, y)
				found = true
			}
		}
	}
	return x
}