CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
SRC := main.cpp

solver: $(SRC)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $(SRC)

.PHONY: clean
clean:
	rm -f solver
