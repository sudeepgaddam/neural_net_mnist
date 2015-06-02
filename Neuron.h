#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
using namespace std;
#define WEIGHT double
class Neuron {
private:
  int numInputs;
  std::vector<WEIGHT>* weights;
  double delta;
  double activation;
  double value;

public:
  Neuron() {}

  Neuron(int inputs) {
    numInputs = inputs;
    // There is an extra weight for the bias input.
    weights = new std::vector<WEIGHT>(numInputs + 1);

    // Setup weights with an initial random value between -1 and 1. There is
    // one weight for each input and an additional bias weight.
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
	double mean =0.0;
    std::normal_distribution<double> distribution (0.0,0.5);


    for (int i = 0; i < weights->size(); i++) {
      (*weights)[i] = distribution(generator);
    }
  }

  ~Neuron() {
    delete weights;
  }

  // Get the corresponding weight.
  WEIGHT getWeight(int n) const {
    return (*weights)[n];
  }

  // Add an update value to a specified input weight.
  void updateWeight(int pos, WEIGHT update) {
    (*weights)[pos] += update;
  }

  // Get the linear combination of inputs to the neuron.
  double getActivation() const {
    return activation;
  }

  // Get the value of the neuron (sigmoid applied to the activation).
  double getValue() const {
    return value;
  }

  // Set the value of the neuron.
  void setValue(double v) {
    value = v;
  }

  // Get the delta value for this neuron.
  double getDelta() const {
    return delta;
  }

  // Set the delta value for this neuron.
  void setDelta(double new_delta) {
    delta = new_delta;
  }

  // Compute and set the linear combination of inputs to the neuron.
  void setActivation(double a) {
    activation = a;
  }

  double printWeights() {
    for (int i = 0; i < weights->size()-1; i++) {
      std::cout << (*weights)[i] << " ";
    }
   // std::cout << endl;
   // std::cout << "activation: " << activation << "value: " << value;
    std::cout << "\n";
  }
};
