CXX = g++
CXXFLAGS = -c -O3 -std=c++1y -I lib/eigen

build = bin/build
dirs = bin/build/layers bin/build/core bin/build/test bin/models bin/logs/fc bin/logs/cn
layers = layers/convolutional_layer.o layers/cross_entropy_cost_layer.o layers/dropout_layer.o layers/fully_connected_layer.o layers/max_pooling_layer.o layers/quadratic_cost_layer.o
core = core/network.o core/summary.o
test = test/mnist.o test/convolutional_test.o test/fully_connected_test.o

all_objects = $(test) $(core) $(layers)
fc_objects = test/fully_connected_test.o test/mnist.o $(core) $(layers)
cn_objects = test/convolutional_test.o test/mnist.o $(core) $(layers)

all: directories fc cn

fc: $(fc_objects)
	g++ -O3 -o fc $(fc_objects:%=bin/build/%)
	
cn: $(cn_objects)
	g++ -O3 -o cn $(cn_objects:%=bin/build/%)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $(build)/$@ $<

# directory stuff
.PHONY: clean clean_all directories $(dirs)

clean:
	rm -R bin/build/*

clean_all:
	rm -R bin
	rm fc cn

directories: $(dirs)

$(dirs):
	mkdir -p $@
