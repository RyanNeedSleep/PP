NVCC := nvcc
NVCCFLAGS := -std=c++17 -O3 -Xcompiler -fopenmp -Wno-deprecated-gpu-targets
CXX := g++
CXXFLAGS := -std=c++17 -O3 -fopenmp
TARGET := hw5
SEQUENTIAL := nbody


.PHONY: all
all: $(TARGET)

.PHONY: hw5
hw5: hw5.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<
.PHONY: seq
seq: samples/nbody.cc
	$(CXX) $(CXXFLAGS) -o $(SEQUENTIAL) $<

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)


