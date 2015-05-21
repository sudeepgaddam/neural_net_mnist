CC = g++ -std=c++11
CLAGS = -Wall
LDFLAGS = $(INCLUDE) -L/usr/local/lib \
          -lopencv_core -lopencv_highgui -lopencv_imgproc
CVFLAGS = `pkg-config --cflags --libs opencv`
ALL = neural

all: $(ALL)

neural: main.o MnistData.o NeuralNet.o
	$(CC) $(CFLAGS) -o neural MnistData.o NeuralNet.o main.o 

main.o: main.cc
	$(CC) $(CFLAGS) -Wno-write-strings $(INCLUDE) -c main.cc

NeuralNet.o: Neuron.h Layer.h NeuralNet.cpp NeuralNet.h
	$(CC) $(CFLAGS) $(INCLUDE) -c  NeuralNet.cpp

MnistData.o: MnistData.cpp MnistData.h
	$(CC) $(CFLAGS) $(INCLUDE) -c MnistData.cpp

.PHONY: clean

clean:
	rm -rf *.o *.gch $(ALL)
