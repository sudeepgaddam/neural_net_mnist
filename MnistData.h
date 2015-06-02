#ifndef MNISTDATA_H
#define MNISTDATA_H

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>

#define Container std::vector
#define Sub std::vector
#define Pixel  double
#define Label  uint8_t


class MnistData {
private:
	//Variables
	Container<Sub<Pixel>> training_images;
    Container<Sub<Pixel>> test_images;
    Container<Label> training_labels;
    Container<Label> test_labels;
    //Rows and coloumns for training and test images
    //Total number of images
	int rows, columns;
    //Functionns
    Container<Sub<Pixel>> readMnistImageFile(const char * path, std::size_t limit );
	uint32_t readHeader(const std::unique_ptr<char[]>& buffer, size_t position);
	Container<Label> readMnistLabelFile(const char* path, std::size_t limit);
	Container<Sub<Pixel>> readTrainingImages(std::size_t limit );
	Container<Sub<Pixel>> readTestImages(std::size_t limit );
	Container<Label> readTrainingLabels(std::size_t limit );
	Container<Label> readTestLabels(std::size_t limit );



public:

  // Consrtuctor Read the mnist data files and store its data
  MnistData(std::size_t training_limit, std::size_t test_limit);
  
	void getPixels(bool training, int img_id, std::vector<Pixel>& input);
	//Print Training images
	void printTrainingImages();
	//Print Training labels
	void printTrainingLabels();
	//Print Test images
	void printTestImages();
	//Print Test labels
	void printTestLabels();
	//Print Both training and test set properties
	void printDataSetSize();
  // Free the image object.
  int getRows();
  int getColoumns();
  int getTrainingSize();
  int getTestSize();
  int getLabel(bool training, int j);
  ~MnistData();
};
#endif
