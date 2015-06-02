#include "MnistData.h"

using namespace std;

uint32_t MnistData::readHeader(const std::unique_ptr<char[]>& buffer, size_t position){
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

Container<Sub<Pixel>> MnistData::readMnistImageFile(const char* path,std::size_t limit = 0){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = readHeader(buffer, 0);

        if(magic != 0x803){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = readHeader(buffer, 1);
            rows = readHeader(buffer, 2);
            columns = readHeader(buffer, 3);

				//cout << "Rows:" << rows << " Coloumns: " << columns;  

            if(size < count * rows * columns + 16){
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            } else {
                //Skip the header
                //Cast to unsigned char is necessary cause signedness of char is
                //platform-specific
                auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

                std::vector<std::vector<Pixel>> images;
                images.reserve(count);

                for(size_t i = 0; i < count; ++i){
                    images.emplace_back(rows * columns);
                    for(size_t j = 0; j < rows * columns; ++j){
                        auto pixel = *image_buffer++;
                        images[i][j] = static_cast<Pixel>(pixel);
                
                    }                   
                }

                return images;
            }
        }
    }

    return {};
}
Container<Label> MnistData::readMnistLabelFile(const char* path, std::size_t limit = 0){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = readHeader(buffer, 0);

        if(magic != 0x801){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = readHeader(buffer, 1);

            if(size < count + 8){
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            } else {
                //Skip the header
                //Cast to unsigned char is necessary cause signedness of char is
                //platform-specific
                auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

                if(limit > 0 && count > limit){
                    count = limit;
                }

                Container<Label> labels(count);

                for(size_t i = 0; i < count; ++i){
                    auto label = *label_buffer++;
                    labels[i] = static_cast<Label>(label);
                }

                return labels;
            }
        }
    }

    return {};
}
Container<Sub<Pixel>> MnistData::readTrainingImages(std::size_t limit){
    return readMnistImageFile("train-images-idx3-ubyte", limit);
}

Container<Sub<Pixel>> MnistData::readTestImages(std::size_t limit){
    return readMnistImageFile("t10k-images-idx3-ubyte", limit);
}

Container<Label> MnistData:: readTrainingLabels(std::size_t limit){
    return readMnistLabelFile("train-labels-idx1-ubyte", limit);
}

Container<Label> MnistData:: readTestLabels(std::size_t limit ){
    return readMnistLabelFile("t10k-labels-idx1-ubyte", limit);
}

MnistData::MnistData(std::size_t training_limit, std::size_t test_limit){

    training_images = readTrainingImages(training_limit);
    training_labels = readTrainingLabels(training_limit);

    test_images = readTestImages(test_limit);
    test_labels = readTestLabels(test_limit);

}
MnistData::  ~MnistData(){
	
}
void MnistData:: printTrainingImages( ){
	
	for (auto a : training_images) {
		for (auto b : a) {
			cout << b << " ";
		}
	cout << endl;
	}
}
void MnistData:: printTrainingLabels( ){
	for (auto a : training_labels) {
		cout << a << " ";
	}
	cout << endl;
}
void MnistData:: printTestImages( ){
	for (auto a : test_images) {
		int line_break = 0;
		for (auto b : a) {
			line_break++;

			cout << b << " ";
             if (line_break >= 28) {
				line_break = 0;
				cout << endl;
			}
		}
	cout << endl;
	}
}
void MnistData:: printTestLabels( ){
	for (auto a : test_labels) {
		cout << a << " ";
	}
	cout << endl;
}

void MnistData:: printDataSetSize( ){
	cout << "In each image, Number of Rows:" << rows << " Coloumns: " << columns <<endl;  
	cout << "Training labels size: " << training_labels.size() << "Testing labels size: " 
	<< test_labels.size() <<endl <<"Training Images size: " << training_images.size() << "Training Images size: " << test_images.size() <<endl;
	cout << endl;
}
int MnistData::getRows() {
	return rows;
}
int MnistData::getColoumns() {
	return columns;
}
int MnistData::getTrainingSize() {
	return training_labels.size();
}
int MnistData::getTestSize() {
	return test_labels.size();
}
void MnistData::getPixels(bool training, int img_id, vector<Pixel>& input){
	if (training) {
		input = training_images[img_id];
	} else {
		input = test_images[img_id];
	}
	for (auto &a : input) {
		if (a>0) {
			
			a=100000.0; 
			
		} else {
			a= 0.0;
		}
	}
}
int MnistData::getLabel(bool training, int j){
	if (training) {
		return training_labels[j];
	}else {
		return test_labels[j];
	}
}
