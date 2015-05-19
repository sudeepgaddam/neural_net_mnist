#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string.h>
#include "include/mnist/mnist_reader.hpp"
using namespace std;
int main(int argc, char *argv[]) {
  srand((unsigned)time(NULL));

  int training = 0, layers = 2, testing = 0;
  double bias = 0, responseThreshold = 1, learningRate = 1;
  int layerHeight = 10;

  // argc is 1 if the command line was given the name of the binary
  // and no additional parameters.
  if (argc == 1) {
    cout << "usage: " << argv[0] << " -t 2 -l 3 -b 12 -a 0 -r 0 -h 5\n"
         << "-t: the number of training samples per digit.\n"
         << "-T: the number of testing samples per digit.\n"
         << "-l: the number of hidden layers; default = 2.\n"
         << "-b: the weight of the bias.\n"
         << "-a: the learning rate for back propagation.\n"
         << "-r: the response threshold for the sigmoid function.\n"
         << "-h: the number of neurons per hidden layer.\n";
    return 0;
  }

  // Process command line arguments.
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-t") == 0) {
      training = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-T") == 0) {
      testing = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-l") == 0) {
      layers = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-b") == 0) {
      bias = atof(argv[++i]);
    } else if (strcmp(argv[i], "-r") == 0) {
      responseThreshold = atof(argv[++i]);
    } else if (strcmp(argv[i], "-a") == 0) {
      learningRate = atof(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0) {
      layerHeight = atoi(argv[++i]);
    }
  }

  if (layers < 0 || training <= 0 || testing <= 0 || responseThreshold <= 0
      || layerHeight <= 0 || learningRate < 0) {
    cout << "Invalid argument specified.\n";
    return 1;
  }

auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

  return 0;
}
