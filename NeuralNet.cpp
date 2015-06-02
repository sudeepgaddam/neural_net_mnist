#include <cmath>
#include <cstdio>
#include <iostream>

#include "NeuralNet.h"

using namespace std;

// Initialize the neural network with the given input parameters, in turn
// initializing each layer with neurons of random weight.
NeuralNet::NeuralNet(int inputs,
                     int outputs,
                     int hiddenLayers,
                     int neuronsPerLayer,
                     double alpha,
                     double threshold) {
  numInputs = inputs;
  numOutputs = outputs;
  numHiddenLayers = hiddenLayers;
  numNeuronsPerLayer = neuronsPerLayer;
    std::vector<double> bias_array;
  bias_array.reserve(hiddenLayers+1);
  learningRate = alpha;
  responseThreshold = threshold;
  layers = new vector<Layer*>(hiddenLayers + 2);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
	double mean =0.0;
    std::normal_distribution<double> distribution (0.0,1.0);
    for (auto &a : bias_array) {
		a = distribution(generator);
	}
  // Initialize each hidden layer.
  (*layers)[0] = new Layer(inputs, 0, bias_array[0]);
  (*layers)[1] = new Layer(neuronsPerLayer, inputs, bias_array[1]);
  (*layers)[hiddenLayers + 1] = new Layer(outputs, neuronsPerLayer, 0.0);
  for (int i = 2; i < layers->size() - 1; i++) {
    (*layers)[i] = new Layer(neuronsPerLayer, neuronsPerLayer, bias_array[i]);
  }
}

NeuralNet::~NeuralNet() {
  for (int i = 0; i < layers->size(); i++) {
    delete (*layers)[i];
  }
  delete layers;
}

// Compute the outputs from a given set of inputs.
void NeuralNet::feedForward(vector<Pixel>& inputs,
                            vector<double>* outputLayer,
                            const double bias) {
  Layer* inputLayer = (*layers)[0];
  for (int i = 0; i < inputLayer->neuronCount(); i++) {
    inputLayer->getNeuron(i)->setValue(static_cast<double>(inputs[i]));
    inputLayer->getNeuron(i)->setActivation(0);
  }
  for (int l = 1; l < numHiddenLayers + 2; l++) {
    Layer *curr = (*layers)[l], *upstream = (*layers)[l-1];
    for (int j = 0; j < curr->neuronCount(); j++) {
      Neuron *n = curr->getNeuron(j);
      double sum = 0;
      for (int i = 0; i < upstream->neuronCount(); i++) {
        sum += n->getWeight(i) * upstream->getNeuron(i)->getValue();
      }
      n->setActivation(sum);
      //cout << " Sum: " << sum << " sigmoid(sum) " << sigmoid(sum);
      n->setValue(sigmoid(sum));
    }
  }

  Layer* lastLayer = (*layers)[numHiddenLayers+1];
  for (int i = 0; i < lastLayer->neuronCount(); i++) {
    (*outputLayer)[i] = lastLayer->getNeuron(i)->getValue();
   // cout << " value:" << lastLayer->getNeuron(i)->getValue();
  }
}

// Back propagate the errors to update the weights.
void NeuralNet::backPropagate(vector<double>* outputs, int teacher) {
	//cout << "teacher" << teacher << endl;
  Layer *outputLayer = (*layers)[numHiddenLayers + 1];
  for (int i = 0; i < outputs->size(); i++) {
    Neuron *n = outputLayer->getNeuron(i);
    double adjusted = -n->getValue();
    if (i == teacher) {
      adjusted += 1;
    }
    n->setDelta(sigmoidPrime(n->getActivation()) * adjusted);
  }

  // Propagate deltas backward from output layer to input layer.
  for (int l = numHiddenLayers; l >= 0; l--) {
    Layer *curr = (*layers)[l], *downstream = (*layers)[l+1];

    for (int i = 0; i < curr->neuronCount(); i++) {
      double sum = 0;
      Neuron *n = curr->getNeuron(i);
      for (int j = 0; j < downstream->neuronCount(); j++) {
        sum += downstream->getNeuron(j)->getWeight(i)
            * downstream->getNeuron(j)->getDelta();
      }
      n->setDelta(sigmoidPrime(n->getActivation()) * sum);
      //cout << " Delta: " << n->getDelta() << "for layer#" << l << "neuron: " << i <<endl; 
      for (int j = 0; j < downstream->neuronCount(); j++) {
        downstream->getNeuron(j)->updateWeight(i,
            learningRate * sigmoid(n->getActivation())
            * downstream->getNeuron(j)->getDelta());
      }
    }
  }
}

// Compute the sigmoid function.
inline double NeuralNet::sigmoid(double activation) {
  return 1.0 / (1.0 + exp(-activation / responseThreshold));
}

// Compute the derivative of the sigmoid function
inline double NeuralNet::sigmoidPrime(double activation) {
  return (sigmoid(activation)*(1-sigmoid(activation)));
}
