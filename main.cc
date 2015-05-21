#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string.h>

#include "MnistData.h"
#include "NeuralNet.h"

using namespace std;
#define ALPHABET_SIZE 10

int process_data(bool training, NeuralNet& nn, double bias, MnistData &mnistdata) {
  int correct = 0;
  int target_size = 6;
  int iterations;
	int rows = mnistdata.getRows();
	int colomns = mnistdata.getColoumns();
  vector<Pixel> inputs(rows*colomns);
  vector<double>* outputs = new vector<double>(ALPHABET_SIZE);
	if (training) {
		iterations = mnistdata.getTrainingSize();
	} else {
		iterations = mnistdata.getTestSize();
	}
	cout << "iterations" << iterations << endl;
  for (int j = 0; j < iterations; j++) {
      delete outputs;
      int label = mnistdata.getLabel(training, j);
      mnistdata.getPixels(training, j, inputs);
      outputs = new vector<double>(ALPHABET_SIZE);
      nn.feedForward(inputs, outputs, bias);

      if (!training) {
        double max_val = 0;
        int max_index = 0;
        for (int k = 0; k < outputs->size(); k++) {
          if ((*outputs)[k] > max_val) {
            max_val = (*outputs)[k];
            max_index = k;
          }
        }
        if (max_index == label) {
          correct++;
        }
      } else {
        nn.backPropagate(outputs, label);
      }
    
  }

  delete outputs;
  return correct;
}


int main(int argc, char *argv[]) {
	MnistData mnistdata(100000, 100000);
	//mnistdata.printTrainingLabels();
	 int training = 0, layers = 2, testing = 0;
  double bias = 0, responseThreshold = 1, learningRate = 1;
  int layerHeight = 100;
	
	  NeuralNet nn(mnistdata.getRows()*mnistdata.getColoumns(),
               ALPHABET_SIZE,
               layers,
               layerHeight,
               learningRate,
               responseThreshold);


  process_data(true, nn, bias, mnistdata);
  int correct = process_data(false, nn, bias, mnistdata);
	cout << "correct " << correct <<endl;

  return 0;
}
/*

int main(int argc, char *argv[]) {
	MnistData mnistdata(100000, 100000);
	mnistdata.printDataSetSize();
  return 0;
}

*/
