CXX = mpicxx
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -g -fopenmp -I/usr/local/cuda-12.8/include
NVCCFLAGS = -std=c++17 -g -Xcompiler -fopenmp -I/usr/local/cuda-12.8/include
LDFLAGS = -lcudart -L/usr/local/cuda-12.8/lib64 -lfaiss -L/home/pichuq/Support_soft/lib/

OBJS = code.o master.o worker.o dataUtils.o  cuda.o

all: mpi_dem

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

cuda.o: cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

mpi_dem: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f *.o mpi_dem
