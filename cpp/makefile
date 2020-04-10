CXX=g++
CXXFLAGS=-Wall -Wextra -Wpedantic -Wconversion -std=c++11 -DNDEBUG -O3 -pthread -fopenmp

all: parallel_weighted_astar

parallel_weighted_astar: *.cpp
	$(CXX) $(CXXFLAGS) *.cpp -o parallel_weighted_astar

clean:
	rm -fr *.o parallel_weighted_astar
